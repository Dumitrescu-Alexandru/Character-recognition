import numpy as np
import pickle
class Saver():
    def __init__(self,img=np.zeros([28,28]),lbl=-1):
        self.image = img
        self.label = lbl
        self.imgs_data = []
        self.lbls_data = []

    def save(self,file_name):
        lbl_file_name = file_name + "future_train_lbls.bin"
        img_file_name = file_name + "future_train_imgs.bin"
        image = np.array(self.image)
        label = np.array(self.label)
        file = open(lbl_file_name, 'ab')
        pickle.dump(self.label, file)
        file.close()
        file = open(img_file_name, 'ab')
        pickle.dump(self.image, file)
        file.close()

    def load(self,file_name):

        images = self.read_images('imagini_salvate/')
        labels = self.read_labels('imagini_salvate/')


        for img in images:
            self.imgs_data.append(img)
        for lbl in labels:
            self.lbls_data.append(lbl)

        self.words_training_number = np.shape(self.imgs_data)[0]


    def read_images(self, file_name):
        img_file_name = file_name + "future_train_imgs.bin"
        fisier_imagini = open(img_file_name, 'rb')
        images = []
        while 1:
            try:
                images.append(pickle.load(fisier_imagini))
            except EOFError:
                break
        fisier_imagini.close()
        return images

    def read_labels(self, file_name):
        lbl_file_name = file_name + "future_train_lbls.bin"
        fisier_labels = open(lbl_file_name, 'rb')
        labels = []
        while 1:
            try:
                labels.append(pickle.load(fisier_labels))
            except EOFError:
                break
        fisier_labels.close()
        return labels

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





