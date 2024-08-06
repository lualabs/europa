import re
from lightning.pytorch.callbacks import Callback

class LogPredictionSamplesCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        if batch_idx == 0:
            scores = outputs["scores"]
            predictions = outputs["predictions"]
            answers = outputs["answers"]
    
            samples_to_log = []
            for pred, answer in zip(predictions, answers):
                pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
                samples_to_log.append([pred, answer])
            print(samples_to_log)
                
            columns = ["pred", "answer"]
            trainer.logger.log_table(key="sample_result", columns=columns, data=samples_to_log)
