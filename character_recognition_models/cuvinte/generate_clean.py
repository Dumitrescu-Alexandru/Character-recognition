from sources.data import Data
from cuvinte.word_generator import data_cleaning
import numpy as np
import pickle

PATH_TO_BINARIES = "C:\\Users\\alex_\\PycharmProjects\\Licenta\\LSTM\\binaries\\"

class Generate(Data):

    smaller_oh = []
    smaller_labels =[]
    smaller_images = []
    small_word_number = []

    small_batch_placement = 0
    batch_placement = 0
    lstm_data_labels_oh = []
    lstm_data_labels = []
    lstm_data_images = []
    words = []
    words_training_number = None        #number of ARtfi. data red
    train_letter_number = None
    letters_train= []
    letters_test = []
    letters_train_labels = []
    letters_test_labels = []
    upper_lower_case = []
    order_upper = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
             14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: []}
    order_lower_thin = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
             14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: []}
    order_lower_medium = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
                        14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [],
                        26: []}
    order_lower_thick = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
                   14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [],
                   26: []}
    def __init__(self):
        super().__init__()
        words = data_cleaning()
        self.words = [item.rstrip() for item in words]
        self.letters_train = self.letter_images
        self.letters_test = self.letters_test
        self.letters_train_labels = self.letter_labels
        self.letters_test_labels = self.letter_labels_test
        file = open(r"../letsarrange.txt")
        self.train_letter_number = int(file.read())
        self.train_letter_number = int(self.train_letter_number)
        self.train_letter_number = self.train_letter_number - 60000
        '''
            THIS WILL BE TRAINED ON THE self.train_letter_number,
            THE NUMBER TO WHICH I HAVE GOT UP TO THIS POINT
        '''
        self.letters_train = self.letters_train[:self.train_letter_number]
        self.letters_train_labels = self.letters_train_labels[:self.train_letter_number]

        file.close()
        file = open(r"../aranjate.txt")
        aux =  file.read()
        self.upper_lower_case = aux.split()
        self.upper_lower_case = list(map(int,self.upper_lower_case))
        file.close()

    def show_image_lettes(self,image):

        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        imgplot = ax.imshow(image, cmap=mpl.cm.gray)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

    def get_in_order(self):
        a = False
        for i in range(self.train_letter_number):
            if self.upper_lower_case[i] == 0:
                self.order_lower_thin[int(self.letter_labels[i])-1].append(self.letters_train[i])
            elif self.upper_lower_case[i] == 1:
                self.order_upper[int(self.letter_labels[i])-1].append(self.letters_train[i])
            elif self.upper_lower_case[i] == 2:
                self.order_lower_medium[int(self.letter_labels[i]) - 1].append(self.letters_train[i])
            else:
                if a == False:
                    if self.upper_lower_case[i] == 3:
                        if int(self.letter_labels[i]) - 1 == 0:
                            a = True
                            self.show_image_lettes(self.letters_train[i])
                else:
                    self.order_lower_thick[int(self.letter_labels[i]) - 1].append(self.letters_train[i])

    def refit(self,image,string_word):
        if (string_word[1] == " " and string_word[2] == " "):
            return image
        elif (string_word[1] != " " and string_word[2] == " "):
            start_from_middle = 28
            i = 28
            j = 28
            while(np.argmax(image[:,i]) == 0):
                i = i - 1
            while(np.argmax(image[:,j]) == 0):
                j = j + 1
            i = i + 1
            close_nr = np.random.randint(2,5)
            image[:,i+close_nr:i+close_nr+28] = image[:,j:j+28]
            image[:,i+close_nr+28:] = image[:,i+close_nr+28:] * 0
            return image

        elif(string_word[1] != " " and string_word[2] != " "):
            first_start = 28
            second_start = 56
            _1Ltr_end = 28
            _2Ltr_begin = 28
            _2Ltr_end = 56
            _3Ltr_begin = 56

            while (np.argmax(image[:, _1Ltr_end]) == 0):
                _1Ltr_end = _1Ltr_end - 1
            while (np.argmax(image[:, _2Ltr_begin]) == 0):
                _2Ltr_begin = _2Ltr_begin + 1
            while (np.argmax(image[:, _2Ltr_end]) == 0):
                _2Ltr_end = _2Ltr_end - 1
            while (np.argmax(image[:, _3Ltr_begin]) == 0):
                _3Ltr_begin = _3Ltr_begin + 1

            close_nr_1 = np.random.randint(2, 5)
            close_nr_2 = np.random.randint(2, 5)

            aux_image = np.zeros((28,28*3))
            ltr_2 = image[:,_2Ltr_begin:_2Ltr_end]
            ltr_2_dim =  _2Ltr_end - _2Ltr_begin + 1
            aux_image[:,:28]  = image[:28,:28]
            aux_image[:,_1Ltr_end+close_nr_1:_1Ltr_end+close_nr_1+ltr_2_dim-1] = ltr_2
            some_nr = _1Ltr_end+close_nr_1+ltr_2_dim+close_nr_2

            if(_2Ltr_begin - _1Ltr_end + _3Ltr_begin - _2Ltr_end   < 10 ):
                return image
            else:
                aux_image[:,some_nr:np.shape(image[:,_3Ltr_begin:])[1]+some_nr] = image[:,_3Ltr_begin:]
            return aux_image

    def create_smaller_data(self,size=3,nr_of_data=1000):
        words = data_cleaning()
        words = [item.rstrip() for item in words]
        words = [word for word in words if len(word) <= size]
        words = [wrd + " " * (size-len(wrd)) for wrd in words]
        self.words = words
        unique_words = np.shape(words)[0]

        lbl_file_name = "words_generate/"+ "lbl_words_{0}_chrs.bin".format(size)
        img_file_name = "words_generate/" + "img_words_{0}_chrs.bin".format(size)
        oh_labels = "words_generate/" + "oh_lbl_{0}_chrs.bin".format(size)
        print(self.words)
        print(unique_words)
        images = []
        labels = []
        labels_oh = []
        select_random_words = np.random.randint(0, unique_words, 5)
        words_selected = []
        for rnd_wrd in select_random_words:
            words_selected.append(words[rnd_wrd])
        words = ['end','or ','hat','hip']
        print(words)
        for i in range(nr_of_data):
            current_label = []
            rnd_nr = np.random.randint(0,3)

            labels.append(words[rnd_nr])
            # generate images so that 80% have medium thickness
            thick_nr = np.random.randint(0,10)
            if thick_nr <=10:
                thick_nr = 1
            elif thick_nr == 8:
                thick_nr = 0
            else:
                thick_nr = 2
            # get the raw image
            img = self.convert_to_images(words[rnd_nr])
            #img = self.refit(img,words[rnd_nr])
            images.append(img)

            #need to append the worked out image
            #images.append(self.convert_to_images(words[rnd_nr]))
            for str in words[rnd_nr]:
                character_oh = np.zeros(27)
                if str == ' ':
                    character_oh[26] = 1
                else:
                    character_oh[ord(str) - 97] = 1
                current_label.append(character_oh)
            labels_oh.append(current_label)
        file = open(lbl_file_name, 'ab')
        pickle.dump(labels, file)
        file.close()
        file = open(img_file_name, 'ab')
        pickle.dump(images, file)
        file.close()
        file = open(oh_labels, 'ab')
        pickle.dump(labels_oh, file)
        file.close()

    def convert_to_images(self,string,first_upper = False,thickness=1):

        l = 0
        new_image = [[None] * len(string) * 28 * 28]
        new_image = np.float32(new_image)
        new_image = np.array(new_image)
        new_image = new_image.reshape(28, (28 * len(string)))
        z = ord(string[0])
        z = z - 97
        c = np.shape(self.order_upper[z])
        max = c[0] - 1
        from_nr = 0
        if first_upper:
            nr = np.random.randint(0, max - 1)
            new_image[0:28, l * 28:(l + 1) * 28] = self.order_upper[z][nr]
            l += 1
            from_nr = 1

        for i in string[from_nr:]:
            if i == " ":
                new_image[0:28, l * 28:(l + 1) * 28] = np.zeros((28,28))
                l += 1
            else:
                z = ord(i)
                z = z - 97
                if thickness == 0:
                    c = np.shape(self.order_lower_thin[z])
                elif thickness == 1:
                    c = np.shape(self.order_lower_medium[z])
                elif thickness == 2:
                    c = np.shape(self.order_lower_thick[z])
                max = c[0] - 1
                nr = np.random.randint(0, max - 1)
                if thickness == 0:
                    new_image[0:28, l * 28:(l + 1) * 28] = self.order_lower_thin[z][nr]
                elif thickness == 1:
                    new_image[0:28, l * 28:(l + 1) * 28] = self.order_lower_medium[z][nr]
                elif thickness == 2:
                    new_image[0:28, l * 28:(l + 1) * 28] = self.order_lower_thick[z][nr]
                l += 1
        #self.show_image_lettes(new_image)
        return new_image

    def append_small_data(self,size=5):
        path = "C:\\Users\\alex_\\PycharmProjects\\Licenta\\cuvinte\\words_generate\\"
        lbl_file_name = path + "lbl_words_{0}_chrs.bin".format(size)
        img_file_name = path + "img_words_{0}_chrs.bin".format(size)
        oh_labels = path + "oh_lbl_{0}_chrs.bin".format(size)

        images_file = open(img_file_name, 'rb')
        images = []
        while 1:
            try:
                images.append(pickle.load(images_file))
            except EOFError:
                break
        images_file.close()
        imgs = []
        for img in images:
            imgs = imgs + img

        label_word_file =open(lbl_file_name, 'rb')
        labels = []
        while 1:
            try:
                labels.append(pickle.load(label_word_file))
            except EOFError:
                break
        label_word_file.close()
        lbls = []
        for lbl in labels:
            lbls = lbls + lbl

        label_oh_file = open(oh_labels, 'rb')
        labels_oh = []
        while 1:
            try:
                labels_oh.append(pickle.load(label_oh_file))
            except EOFError:
                break
        label_word_file.close()
        lbls_oh = []
        for lbl_oh in labels_oh:
            lbls_oh = lbls_oh + lbl_oh
        self.small_word_number = np.shape(imgs)[0]
        self.smaller_images = imgs
        self.smaller_labels = lbls
        self.smaller_oh = lbls_oh
a = Generate()

a.get_in_order()



a.create_smaller_data(nr_of_data=30000)
