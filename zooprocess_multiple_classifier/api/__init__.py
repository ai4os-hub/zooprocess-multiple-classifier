from webargs import fields

schema = {
    "score": fields.Float(
        required=True,
        metadata={
            'description': "in [0,1]: the probability for the image to be a multiple. A natural threshold to classify it as multiple is 0.5 but lowering this threshold can increase the recall of multiples, at the expense of precision."
        }
    )
}
