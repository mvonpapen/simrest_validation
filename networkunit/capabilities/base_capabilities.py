import sciunit


class ProducesSample(sciunit.Capability):
    """
    Here general porperties and checks of capabilities can be defined which
    produce a sample of a property.
    """

    def produce_sample(self):
        return self.unimplemented()