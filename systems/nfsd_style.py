from dataclasses import dataclass

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

import threestudio
import wandb
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import Any, Dict


def get_blip_text_from_image(image_path, resolution=512, device="cuda"):
    """Extract image caption from an image using BLIP model."""
    style_image = Image.open(image_path).resize((resolution, resolution), resample=Image.BICUBIC)

    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(device)
    blip_model.eval()

    inputs = blip_processor(images=style_image, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip_model.generate(**inputs)
    text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    del blip_processor, blip_model

    return text


@threestudio.register("nfsd-style-system")
class DreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        pass

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        """Initialize guidance models and prompt processors."""
        super().on_fit_start()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # Extract style description from the reference image
        style_prompt = get_blip_text_from_image(
            image_path=self.cfg.guidance["style_image_path"],
            resolution=self.cfg.guidance["style_image_size"],
        )

        # Create prompt processors for different components
        # 1. Style prompt procfessor
        prompt_processor_style_cfg = self.cfg.prompt_processor.copy()
        prompt_processor_style_cfg.prompt = style_prompt
        prompt_processor_style = threestudio.find(self.cfg.prompt_processor_type)(prompt_processor_style_cfg)
        self.prompt_utils_style = prompt_processor_style()

        # 2. Negative prompt processor
        prompt_processor_neg_cfg = self.cfg.prompt_processor.copy()
        prompt_processor_neg_cfg.prompt = (
            "unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy"
        )
        prompt_processor_neg = threestudio.find(self.cfg.prompt_processor_type)(prompt_processor_neg_cfg)
        self.prompt_utils_neg = prompt_processor_neg()

        # 3. Object prompt processor
        prompt_processor_object = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.prompt_utils_object = prompt_processor_object()
        wandb.log({"object_prompt": self.cfg.prompt_processor["prompt"]})

        # 4. Styled object prompt processor (combining style and object)
        # Determine how to combine the style and object prompts
        if self.cfg.guidance["style_blip_object"] is None:
            prompt = self.cfg.prompt_processor["prompt"] + " in style of " + style_prompt
        else:
            assert self.cfg.guidance["style_blip_object"] in style_prompt, (
                f"Object '{self.cfg.guidance['style_blip_object']}' not found in the blip style prompt: '{style_prompt}'."
            )
            prompt = style_prompt.replace(
                self.cfg.guidance["style_blip_object"], self.cfg.prompt_processor["prompt"]
            )

        self.cfg.prompt_processor["prompt"] = prompt
        prompt_processor_styled_object = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.prompt_utils_styled_object = prompt_processor_styled_object()

        # Log prompts to wandb for tracking
        wandb.log({"style_prompt": style_prompt})
        wandb.log({"styled_object_prompt": prompt})

    def training_step(self, batch, batch_idx):
        out = self(batch)
        guidance_out = self.guidance(
            **batch,
            rgb=out["comp_rgb"],
            prompt_utils_styled_object=self.prompt_utils_styled_object,
            prompt_utils_style=self.prompt_utils_style,
            prompt_utils_object=self.prompt_utils_object,
            prompt_utils_neg=self.prompt_utils_neg,
            rgb_as_latents=False,
            global_step=self.global_step,
        )

        loss = 0.0

        for name, value in guidance_out.items():
            if not (type(value) is torch.Tensor and value.numel() > 1):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                self.log(f"train_lambda/{name}", value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")]))

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError("Normal is required for orientation loss, no normal is found in the output.")
            loss_orient = (out["weights"].detach() * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2).sum() / (
                out["opacity"] > 0
            ).sum()
            self.log("train/loss_orient", loss_orient)
            self.log("train_lambda/loss_orient", loss_orient * self.C(self.cfg.loss.lambda_orient))
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        self.log("train_lambda/loss_sparsity", loss_sparsity * self.C(self.cfg.loss.lambda_sparsity))
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        self.log("train_lambda/loss_opaque", loss_opaque * self.C(self.cfg.loss.lambda_opaque))
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z-variance loss proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if "z_variance" in out and "lambda_z_variance" in self.cfg.loss:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            self.log("train_lambda/loss_z_variance", loss_z_variance * self.C(self.cfg.loss.lambda_z_variance))
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        out_white_bg = self.renderer(**batch, bg_color=torch.Tensor([1, 1, 1]).to(get_device()))

        normals_on_white = out_white_bg["comp_normal"][0]
        black_mask = (normals_on_white < 0.05).all(dim=2).unsqueeze(2).repeat(1, 1, 3)

        normals_on_white = torch.where(black_mask, torch.ones_like(normals_on_white), normals_on_white)

        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out_white_bg["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": normals_on_white,
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }
            ]
            + [
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-val",
            "",
            f"it{self.true_global_step}\-(\d+).png",
            save_format="mp4",
            name="val",
            step=self.true_global_step,
        )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        out_white_bg = self.renderer(**batch, bg_color=torch.Tensor([1, 1, 1]).to(get_device()))

        normals_on_white = out_white_bg["comp_normal"][0]
        black_mask = (normals_on_white < 0.05).all(dim=2).unsqueeze(2).repeat(1, 1, 3)

        normals_on_white = torch.where(black_mask, torch.ones_like(normals_on_white), normals_on_white)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out_white_bg["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": normals_on_white,
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }
            ]
            + [
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
