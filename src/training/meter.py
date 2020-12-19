import numpy as np

from utils.metrics import boxes_f1_score, detection_ratio


class NFLMeter:
    def __init__(self, num_classes=1):
        self.reset()
        self.num_classes = num_classes

    def update_preds(self, preds, vids, frames):
        for i in range(len(preds)):
            scores = preds[i].detach().cpu().numpy()[:, 4:]

            boxes = preds[i].detach().cpu().numpy()[:, :4]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

            frame = int(frames[i]) * np.ones((len(boxes), 1))
            pred = np.concatenate((frame, boxes), 1)

            self.pred_videos += [vids[i]] * len(boxes)
            self.preds.append(pred)
            self.pred_scores.append(scores)

    def update_truths(self, truths, labels, vids, frames):
        for i in range(len(truths)):
            boxes = truths[i].cpu().numpy()
            boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

            frame = frames[i] * np.ones((len(boxes), 1))
            truth = np.concatenate((frame, boxes), 1)

            self.truth_videos += [vids[i]] * len(boxes)
            self.truths.append(truth)
            self.true_labels.append(labels[i].cpu().numpy())

    def update(self, batch, preds):
        boxes, labels, vids, frames = batch[1:5]

        self.update_preds(preds, vids, frames)
        self.update_truths(boxes, labels, vids, frames)

    def concat(self):
        self.truths = np.concatenate(self.truths)
        self.preds = np.concatenate(self.preds)

        self.pred_scores = np.concatenate(self.pred_scores)
        self.true_labels = np.concatenate(self.true_labels)

        self.pred_videos = np.array(self.pred_videos)
        self.truth_videos = np.array(self.truth_videos)

        self.concatenated = True

    def compute_scores(self):
        """
        Uses 3 thresholding strategies
        """
        if self.num_classes == 2:
            score_1 = self.competition_metric(
                helmet_threshold=0.5,
                impact_threshold=0.5,
            )[1]

            score_2 = self.competition_metric(
                helmet_threshold=0.5,
                impact_threshold_ratio=0.5,
            )[1]

            score_3 = self.competition_metric(
                impact_threshold=0.5,
            )[1]
        else:
            score_1 = self.detection_metric(threshold=0.1)
            score_2 = self.detection_metric(threshold=0.25)
            score_3 = self.detection_metric(threshold=0.5)

        return score_1, score_2, score_3

    def get_helmets(
        self,
        helmet_threshold=0.,
        impact_threshold_ratio=0.,
        impact_threshold=0.
    ):
        if not self.concatenated:
            self.concat()

        preds_helmet, truths_helmet = [], []
        preds_impact, truths_impact = [], []

        for vid in np.unique(self.truth_videos):
            preds = self.preds[self.pred_videos == vid]
            scores = self.pred_scores[self.pred_videos == vid]
            truth = self.truths[self.truth_videos == vid]
            true_label = self.true_labels[self.truth_videos == vid]

            helmets = scores[:, 0] > helmet_threshold
            scores_helmet = scores[helmets]

            if self.num_classes > 1:
                if impact_threshold_ratio:
                    ratio = scores_helmet[:, 1] / scores_helmet[:, 0]
                    impacts = ratio > impact_threshold_ratio
                    preds_impact.append(preds[helmets][impacts])
                elif impact_threshold:
                    impacts = scores_helmet[:, 1] > impact_threshold
                    preds_impact.append(preds[helmets][impacts])

            preds_helmet.append(preds[helmets])
            truths_helmet.append(truth)
            truths_impact.append(truth[true_label == 2])

        return preds_helmet, truths_helmet, preds_impact, truths_impact

    def competition_metric(
        self,
        compute_score_helmet=False,
        helmet_threshold=0.,
        impact_threshold_ratio=0.,
        impact_threshold=0.
    ):
        if not self.concatenated:
            self.concat()

        assert len(self.truth_videos) == len(self.truths)
        assert len(self.pred_videos) == len(self.preds)

        if not len(self.preds):
            self.score_helmet, self.score_impact = 0, 0
            return self.score_helmet, self.score_impact

        preds_helmet, truths_helmet, preds_impact, truths_impact = self.get_helmets(
            helmet_threshold=helmet_threshold,
            impact_threshold_ratio=impact_threshold_ratio,
            impact_threshold=impact_threshold,
        )

        if compute_score_helmet:
            self.score_helmet = boxes_f1_score(preds_helmet, truths_helmet)
        self.score_impact = boxes_f1_score(preds_impact, truths_impact)

        return self.score_helmet, self.score_impact

    def detection_metric(self, threshold=0.5):
        preds_helmet, _, _, truths_impact = self.get_helmets(helmet_threshold=threshold)
        return detection_ratio(preds_helmet, truths_impact)

    def reset(self):
        self.truths = []
        self.true_labels = []

        self.preds = []
        self.pred_scores = []

        self.pred_videos = []
        self.truth_videos = []

        self.score_helmet, self.score_impact = 0, 0
        self.concatenated = False
