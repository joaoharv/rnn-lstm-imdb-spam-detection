import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="cp2")
def padronizar(textos):
    """Minúsculas; !?%$@/ viram tokens próprios; apóstrofos internos e acentos são mantidos; demais pontuações viram espaço."""
    t = tf.strings.lower(textos, encoding="utf-8")
    t = tf.strings.regex_replace(t, r"<br\s*/?>", " ")
    t = tf.strings.regex_replace(t, "[’‘`´]", "'")
    t = tf.strings.regex_replace(t, r"([!?%$@/])", r" \1 ")
    t = tf.strings.regex_replace(t, r"[^\pL\pN\s'_!?%$@/]", " ")
    return tf.strings.regex_replace(t, r"(^|\s)'+|'+(\s|$)", " ")
