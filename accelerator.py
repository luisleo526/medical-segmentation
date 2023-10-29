from accelerate import Accelerator

from utils import dice_score, iou_score


class MedicalAccelerator(Accelerator):
    def __int__(self, targets, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_metrics(self, y_pred, y_true, loss, metrics, targets):
        batch_size = y_pred.shape[0]

        dice = dice_score(y_pred, y_true, len(targets))
        iou = iou_score(y_pred, y_true, len(targets))

        loss, dice, iou = self.gather((loss.detach(), dice, iou))

        metrics['loss'].append(loss.cpu().mean(dim=0), batch_size)
        metrics['dice'].append(dice.cpu().mean(dim=0), batch_size)
        metrics['iou'].append(iou.cpu().mean(dim=0), batch_size)
