import config.config as configuration

config = configuration.Configuration().validate()

import preprocessing
import processing
import validation

if __name__ == '__main__':
    #preprocessing.preprocess_all(config)

    #processing.process_all(config)

    validation.validate_all(config)
