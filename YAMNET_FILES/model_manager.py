"""Module ModelManager with ModelManager Class """

import os
import glob
import random
from datetime import datetime
import warnings
import sklearn
# Model Maker for the Audio domain needs TensorFlow 2.5 to work.
import tensorflow as tf
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile

DATA_DIR = './dataset/split_directories/'
MODELS_PATH = './tf_lite_models'
# this resolves permissions problems
os.environ['TFHUB_CACHE_DIR'] = './tf_cache'
# this disable printing of info messages
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set K_FOLD_EVALUATION to True to do the k fold evaluation,
# in k fold evaluation 'mode' the only methods you can use are
# model_10_fold_cross_validation, model_3_fold_cross_validation,
# all the static methods and model_create_train (but this method
# should not be used).
K_FOLD_EVALUATION = True


class ModelManager:
    """Class for providing the methods for managing the tensorflow model"""

    def __init__(self):
        """
        When using Model Maker for audio, you have to start with a model spec.
        This is the base model that your new model will extract information
        to learn about the new classes. It also affects how the dataset will
        be transformed to respect the models spec parameters like: sample rate,
        number of channels.
        frame_length is to decide how long each training sample is.
        frame_step is to decide how far apart are the training samples.
        In this case, the ith sample will start at EXPECTED_WAVEFORM_LENGTH * 6s
        after the (i-1)th sample.
        The reason to set these values is to work around some limitation
        in real world dataset.
        For example, in the dataset, some activities don't 'make sound'
        all the time, with noises in between.
        Having a long frame would help capture the sound of the activity,
        but setting it too long will reduce the number of samples for training.
        """
        self._spec = audio_classifier.YamNetSpec(
            keep_yamnet_and_custom_heads=True,
            frame_step=3 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
            frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)
        # Number of samples per training step.
        self._batch_size = 8
        self._epochs = 100
        # Model Maker has the API to load the data from a folder and have it in
        # the expected format for the model spec.
        # dataloader randomly shuffle the dataset
        # The cache=True is important to make training later faster but it will
        # also require more RAM to hold the data.
        # For the HAR dataset that is not a problem since it is less than a GB.
        self._id_to_name = {0: "male", 1: "female"}
        self._num_classes = 2 # real number (not n-1)
        tmp = audio_classifier.DataLoader.from_folder(
            self._spec, os.path.join(DATA_DIR, 'train'), cache=True)
        self._test_data = False
        if not K_FOLD_EVALUATION:
            self._train_data, self._validation_data = tmp.split(0.8)
        else:
            self._train_data = False
            self._validation_data = False
            self._index_to_label = tmp.index_to_label
            print(self._index_to_label)
        self._model = False
        if not K_FOLD_EVALUATION:
            self.model_create_train()

    @staticmethod
    def show_tf_mm_versions():
        """Prints TensorFlow and Model Maker versions"""
        print(f"TensorFlow Version: {tf.__version__}")
        print(f"Model Maker Version: {mm.__version__}")

    @staticmethod
    def get_random_audio_file():
        """Returns a random audio path from the dataset"""
        test_files = os.path.abspath(os.path.join(DATA_DIR, 'test/*/*'))
        test_list = glob.glob(test_files)
        random_audio_path = random.choice(test_list)
        return random_audio_path

    @staticmethod
    def show_audio_data(audio_path):
        """Shows an audio file from the path in input that must be
        the path of an audio file"""
        audio_code = audio_path.split('/')[-2]
        print(f'Audio class: {audio_code}')

    def show_confusion_matrix(self, confusion):
        """Computes the normalized confusion matrix."""
        confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
        axis_labels = self._test_data.index_to_label
        a_x = sns.heatmap( # pylint: disable=unused-variable
            confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
            cmap='Blues', annot=True, fmt='.2f', square=True)
        plt.title("Confusion matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    def model_create_train(self):
        """Creates a model (audio_classifier) and trains it with the data already loaded."""
        print('\nCreating and Training the model\n')
        self._model = audio_classifier.create(
            self._train_data,
            self._spec,
            self._validation_data,
            batch_size=self._batch_size,
            epochs=self._epochs)
        print("\nDONE.\n")

    def model_evaluate(self):
        """Evaluates the model"""
        print('\nEvaluating the model\n')
        if not K_FOLD_EVALUATION:
            self._test_data = audio_classifier.DataLoader.from_folder(
                self._spec, os.path.join(DATA_DIR, 'test'), cache=True)
        evaluations = self._model.evaluate(self._test_data)
        confusion_matrix, truth, predicated = self._model.confusion_matrix(self._test_data)
        print(confusion_matrix)
        truth_arr = []
        predicated_arr = []
        for v in truth:
            truth_arr.append(v.numpy())
        for v in predicated:
            predicated_arr.append(v.numpy())
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(truth_arr, predicated_arr)
        evaluations.append(balanced_accuracy)
        fp = []
        tn = []
        fn = []
        # True Positives are the diagonal elements
        tp = np.diag(confusion_matrix.numpy())
        # False Positives are the sum of the respective column, minus the diagonal element
        for i in range(self._num_classes):
            fp.append(sum(confusion_matrix.numpy()[:, i]) - confusion_matrix.numpy()[i, i])
        # False Negatives are the sum of the respective row, minus the diagonal element
        for i in range(self._num_classes):
            fn.append(sum(confusion_matrix.numpy()[i, :]) - confusion_matrix.numpy()[i, i])
        # say class 0: it means all the samples that have been correctly identified as not being 0.
        # So, essentially what we should do is remove the corresponding row & column from the
        # confusion matrix, and then sum up all the remaining elements
        for i in range(self._num_classes):
            temp = np.delete(confusion_matrix.numpy(), i, 0)  # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            tn.append(sum(sum(temp)))
        '''
        l = len(predicated_arr)
        for i in range(self._num_classes):
            test = tp[i] + fp[i] + fn[i] + tn[i]
            print(f"test_num_dati: l:{l}, test:{test}, result:", test==l)
        print("confusion matrix: ")
        print(confusion_matrix)
        '''
        if not K_FOLD_EVALUATION:
            self.show_confusion_matrix(confusion_matrix.numpy())
        print("\nDONE.\n")
        return evaluations, confusion_matrix, tp, fp, tn, fn

    def model_export(self):
        """exports the model"""
        assert not K_FOLD_EVALUATION
        print(f'Exporting the TFLite model to {MODELS_PATH}')
        self._model.export(MODELS_PATH, tflite_filename='har_model.tflite')
        # You can also export the SavedModel version for
        # serving or using on a Python environment.
        # self._model.export(MODELS_PATH, export_format=
        #                   [mm.ExportFormat.SAVED_MODEL, mm.ExportFormat.LABEL])

    def k_fold_evaluation(self, num_folds, test_data, training_data):
        """K-fold Cross Validation"""
        accuracy_per_fold = []
        loss_per_fold = []
        date_time_obj = datetime.now()
        datetime_str = f"{num_folds}_fold_average_score_" + str(date_time_obj.hour) + "." + \
                       str(date_time_obj.minute) + "." + str(date_time_obj.second) + ".txt"
        a_file = open(datetime_str, "w", encoding="utf8")
        fold_no = 1
        confusion_matrix_total = None
        sensitivity_per_fold = []
        specificity_per_fold = []
        balanced_accuracy_per_fold = []
        accuracy_per_fold_per_category = []
        while fold_no <= num_folds:
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            # Data prep
            self._train_data, self._validation_data = training_data[fold_no - 1].split(0.9)
            self._test_data = test_data[fold_no - 1]
            # Training
            self.model_create_train()
            # Generate generalization metrics
            scores, confusion_matrix, tp, fp, tn, fn = self.model_evaluate()
            print(confusion_matrix)
            if fold_no == 1:
                confusion_matrix_total = confusion_matrix
            else:
                confusion_matrix_total = tf.add(confusion_matrix_total, confusion_matrix)
            sensitivity = np.sum(tp) / (np.sum(tp) + np.sum(fn))
            specificity = np.sum(tn) / (np.sum(tn) + np.sum(fp))
            print("specificity: ", specificity)
            sensitivity_per_fold.append(sensitivity)
            specificity_per_fold.append(specificity)
            balanced_accuracy_per_fold.append(scores[2])
            print(f'Score for fold {fold_no}: loss:{scores[0]}; accuracy:{scores[1]}; '
                  f'sensitivity:{sensitivity}; specificity:{specificity}')
            accuracy_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            tmp = []
            for i in range(0, self._num_classes):
                tmp_accuracy = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])
                tmp.append(tmp_accuracy)
            accuracy_per_fold_per_category.append(tmp)
            # Increase fold number
            fold_no = fold_no + 1

        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        print('------------------------------------------------------------------------',
              file=a_file)
        print('Score per fold', file=a_file)
        for i in range(0, len(accuracy_per_fold)):
            print('------------------------------------------------------------------------',
                  file=a_file)
            print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: '
                  f'{accuracy_per_fold[i]}% - Sensitivity: {sensitivity_per_fold[i]}'
                  f' - Specificity: {specificity_per_fold[i]}'
                  f' - Balanced Accuracy: {balanced_accuracy_per_fold[i]}',
                  file=a_file)
            print('------------------------------------------------------------------------')
            print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: '
                  f'{accuracy_per_fold[i]}% - Sensitivity: {sensitivity_per_fold[i]}'
                  f' - Specificity: {specificity_per_fold[i]}'
                  f' - Balanced Accuracy: {balanced_accuracy_per_fold[i]}')

        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(accuracy_per_fold)} (+- {np.std(accuracy_per_fold)})')
        print(f'> Balanced Accuracy: {np.mean(balanced_accuracy_per_fold)} '
              f'(+- {np.std(balanced_accuracy_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}'
              f'\n> Sensitivity: {np.mean(sensitivity_per_fold)}'
              f'\n> Specificity: {np.mean(specificity_per_fold)}'
              f'\n> Standard deviation per class: ')
        for i in range(0, self._num_classes):
            tmp = []
            for j in range(0, num_folds):
                tmp.append(accuracy_per_fold_per_category[j][i])
            print(f'>    {self._id_to_name[i]}: +-{np.std(tmp)}')
        print(f'\nConfusion Matrix: \n {confusion_matrix_total}')
        print('------------------------------------------------------------------------')
        print('------------------------------------------------------------------------',
              file=a_file)
        print('Average scores for all folds:', file=a_file)
        print(f'> Accuracy: {np.mean(accuracy_per_fold)} (+- {np.std(accuracy_per_fold)})',
              file=a_file)
        print(f'> Balanced Accuracy: {np.mean(balanced_accuracy_per_fold)}'
              f' (+- {np.std(balanced_accuracy_per_fold)})',
              file=a_file)
        print(f'> Loss: {np.mean(loss_per_fold)}'
              f'\n> Sensitivity: {np.mean(sensitivity_per_fold)}'
              f'\n> Specificity: {np.mean(specificity_per_fold)}'
              f'\n> Balanced Accuracy: {np.mean(balanced_accuracy_per_fold)} '
              f'(+- {np.std(balanced_accuracy_per_fold)})'
              f'\n> Standard deviation per class: ', file=a_file)
        for i in range(0, self._num_classes):
            tmp = []
            for j in range(0, num_folds):
                tmp.append(accuracy_per_fold_per_category[j][i])
            print(f'>    {self._id_to_name[i]}: +-{np.std(tmp)}', file=a_file)
        print('------------------------------------------------------------------------',
              file=a_file)
        a_file.close()
        self.show_confusion_matrix(confusion_matrix_total.numpy())

    def model_3_fold_cross_validation(self):
        """ implements 3 fold cross validation
        Using same method implemented to test on base dataset
        (splitting 70-30 the dataset, and not 66,66 and 33,33)
        """
        assert K_FOLD_EVALUATION
        num_folds = 3
        test_data = [0] * 3
        training_data = [0] * 3
        # slicing data (90% training -> split in train and validation) (10% test)
        # slicing
        load_tmp = audio_classifier.DataLoader.from_folder(
            self._spec, os.path.join(DATA_DIR, 'train'), cache=True, shuffle=True)

        training_data[0], test_data[0] = load_tmp.split(0.7)
        test_data[2], training_data[2] = load_tmp.split(0.3)
        tmp_1, tmp_2 = load_tmp.split(0.65)
        tmp_3, test_data[1] = tmp_1.split(0.54)
        dataset = tmp_3._dataset.concatenate(tmp_2._dataset)
        size = tmp_3._size + tmp_2._size
        training_data[1] = audio_classifier.DataLoader(dataset, size,
                                                       self._index_to_label,
                                                       self._spec, cache=True)
        self.k_fold_evaluation(num_folds, test_data, training_data)

    def model_10_fold_cross_validation(self):
        """implements 10 fold cross validation,
        prints results on command line and on file"""
        assert K_FOLD_EVALUATION
        num_folds = 10
        test_data = [0] * 10
        training_data = [0] * 10
        # slicing data (90% training -> split in train and validation) (10% test)
        # slicing x test data
        load_tmp = audio_classifier.DataLoader.from_folder(
            self._spec, os.path.join(DATA_DIR, 'train'), cache=True, shuffle=True)
        tmp80, tmp20 = load_tmp.split(0.8)
        test_data[1], test_data[0] = tmp20.split(0.5)
        tmp40_1, tmp40_2 = tmp80.split(0.5)
        tmp20_1_1, tmp20_1_2 = tmp40_1.split(0.5)
        tmp20_2_1, tmp20_2_2 = tmp40_2.split(0.5)
        test_data[9], test_data[8] = tmp20_1_1.split(0.5)
        test_data[7], test_data[6] = tmp20_1_2.split(0.5)
        test_data[5], test_data[4] = tmp20_2_1.split(0.5)
        test_data[3], test_data[2] = tmp20_2_2.split(0.5)
        # slicing x training data
        # training_data 0
        training_data[0], discard = load_tmp.split(0.9) # pylint: disable=unused-variable
        # training_data 9
        discard, training_data[9] = load_tmp.split(0.1)
        # Loop for training data from 1 to 8
        dictionary_percentage = {1: [0.8, 0.9], 2: [0.7, 0.8], 3: [0.6, 0.7],
                                 4: [0.5, 0.6], 5: [0.4, 0.5], 6: [0.3, 0.4],
                                 7: [0.2, 0.3], 8: [0.1, 0.2]}
        for i in range(1, 9):
            tmp_1, discard = load_tmp.split(dictionary_percentage[i][0])
            discard, tmp_2 = load_tmp.split(dictionary_percentage[i][1])
            dataset = tmp_1._dataset.concatenate(tmp_2._dataset)
            size = tmp_1._size + tmp_2._size
            training_data[i] = audio_classifier.DataLoader(dataset, size,
                                                           self._index_to_label,
                                                           self._spec, cache=True)
        self.k_fold_evaluation(num_folds, test_data, training_data)

    def model_testing_on_single_file(self):
        """TESTING ON SINGLE FILE"""
        assert not K_FOLD_EVALUATION
        serving_model = self._model.create_serving_model()
        print(f'Model\'s input shape and type: {serving_model.inputs}')
        print(f'Model\'s output shape and type: {serving_model.outputs}')
        random_audio = self.get_random_audio_file()
        self.show_audio_data(random_audio)

        # The model created has a fixed input window.
        # For a given audio file, you'll have to split it
        # in windows of data of the expected size.
        # The last window might need to be filled with zeros.
        sample_rate, audio_data = wavfile.read(random_audio) # pylint: disable=unused-variable
        audio_data = np.array(audio_data) / tf.int16.max
        input_size = serving_model.input_shape[1]
        splitted_audio_data = tf.signal.frame(audio_data, input_size, input_size,
                                              pad_end=True, pad_value=0)
        print(f'Test audio path: {random_audio}')
        print(f'Original size of the audio data: {len(audio_data)}')
        print(f'Number of windows for inference: {len(splitted_audio_data)}')
        print(random_audio)
        # You'll loop over all the splitted audio and apply the model for each one of them.
        # The model you've just trained has 2 outputs: The original
        # YAMNet's output and the one you've just trained.
        # The real world environment is complicated so we can use the
        # YAMNet's output to filter out non relevant audio,
        # (this can also be done in Android studio when you work with the model)
        # Below both outpus are printed to make it easier to understand their relation.
        results = []
        print('Result of the window ith:  your model class -> score,  (spec class -> score)')
        for i, data in enumerate(splitted_audio_data):
            yamnet_output, inference = serving_model(data)
            results.append(inference[0].numpy())
            result_index = tf.argmax(inference[0])
            spec_result_index = tf.argmax(yamnet_output[0])
            result_str = f'Result of the window {i}: ' \
                         f'\t{self._test_data.index_to_label[result_index]} -> ' \
                         f'{inference[0][result_index].numpy():.3f}, ' \
                         f'\t({self._spec._yamnet_labels()[spec_result_index]} -> ' \
                         f'{yamnet_output[0][spec_result_index]:.3f})'
            print(result_str)
        results_np = np.array(results)
        mean_results = results_np.mean(axis=0)
        result_index = mean_results.argmax()
        print(f'Mean result: {self._test_data.index_to_label[result_index]} '
              f'-> {mean_results[result_index]}')
