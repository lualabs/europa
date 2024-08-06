import lightning as L
import torch
import re
from nltk import edit_distance
import numpy as np
from europa.config import EuropaConfig

class EuropaModule(L.LightningModule):
    def __init__(
            self,
            cfg: EuropaConfig,
        ):
        super().__init__()
        hparams = cfg.model_dump()
        self.hparams.update(hparams)
        self.config = cfg
        self.processor = cfg.model.processor
        self.model = cfg.model.load_model()
        self.max_length = cfg.data.max_length
        self.batch_size = cfg.data.batch_size

    def training_step(self, batch, batch_idx):

        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                pixel_values=pixel_values,
                                labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, max_new_tokens=self.max_length)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

        self.log("val_edit_distance", np.mean(scores), on_step=False, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        return {"scores": scores, "predictions": predictions, "answers": answers}

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.train.lr)

        return optimizer
    
    def generate(self, image, prompt="extract JSON.", max_new_tokens=None, **kwargs):
        """
        Generate text based on input image and prompt.
        
        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            prompt (str, optional): Text prompt to guide generation. Defaults to PROMPT.
            max_new_tokens (int, optional): Maximum number of new tokens to generate.
        
        Returns:
            str: The generated text.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_length

        # Prepare inputs
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)

        # Generate
        self.model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                **kwargs
            )

        # Decode and return the generated text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()