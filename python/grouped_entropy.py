import tensorflow as tf
# Custom Loss Functions
@tf.function
def cross_entropy_t(
    labels,
    predictions,
    sample_weight=None,
    focal_gamma=None,
    class_weight=None,
    epsilon=1e-7,
):
    # get true-negative component
    predictions = tf.clip_by_value(predictions, epsilon, 1 - epsilon)
    tn = labels * tf.math.log(predictions)
    # focal loss?
    if focal_gamma is not None:
        tn *= (1 - predictions) ** focal_gamma
    # convert into loss
    loss = -tn
    # apply class weights
    if class_weight is not None:
        loss *= class_weight
    # apply sample weights
    if sample_weight is not None:
        loss *= sample_weight[:, tf.newaxis]

    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

@tf.function
def grouped_cross_entropy_t(
    labels,
    predictions,
    sample_weight=None,
    group_ids=None,
    focal_gamma=None,
    class_weight=None,
    epsilon=1e-7,
    std_xent=True,
):
    loss_terms = []
    if std_xent:
        loss_terms.append(
            cross_entropy_t(
                labels,
                predictions,
                sample_weight=sample_weight,
                focal_gamma=focal_gamma,
                class_weight=class_weight,
                epsilon=epsilon,
            )
        )
    if group_ids:
        # create grouped labels and predictions
        labels_grouped, predictions_grouped = [], []
        for _, ids in group_ids:
            labels_grouped.append(
                tf.reduce_sum(tf.gather(labels, ids, axis=-1), axis=-1, keepdims=True)
            )
            predictions_grouped.append(
                tf.reduce_sum(tf.gather(predictions, ids, axis=-1), axis=-1, keepdims=True)
            )

        labels_grouped = (
            tf.concat(labels_grouped, axis=-1) if len(labels_grouped) > 1 else labels_grouped[0]
        )
        predictions_grouped = (
            tf.concat(predictions_grouped, axis=-1)
            if len(predictions_grouped) > 1
            else predictions_grouped[0]
        )
        loss_terms.append(
            cross_entropy_t(
                labels_grouped,
                predictions_grouped,
                sample_weight=sample_weight,
                focal_gamma=focal_gamma,
                class_weight=tf.constant([w for w, _ in group_ids], tf.float32),
                epsilon=epsilon,
            )
        )
    return sum(loss_terms) / len(loss_terms)



class GroupedXEnt(tf.keras.losses.Loss):
    def __init__(
        self,
        group_ids=None,
        focal_gamma=None,
        class_weight=None,
        epsilon=1e-7,
        std_xent=True,
        *args,
        **kwargs
    ):
        super(GroupedXEnt, self).__init__(*args, **kwargs)
        self.group_ids = group_ids
        self.focal_gamma = focal_gamma
        self.class_weight = class_weight
        self.epsilon = epsilon
        self.std_xent = std_xent

    def call(self, y_true, y_pred, sample_weight=None):
        return grouped_cross_entropy_t(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            group_ids=self.group_ids,
            focal_gamma=self.focal_gamma,
            class_weight=self.class_weight,
            epsilon=self.epsilon,
            std_xent=self.std_xent
        )


