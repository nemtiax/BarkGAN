class Trainer(object):
    def __init__(self):

    def load_data(self):
        self.data_files = glob(os.path.join("./data", "tiles", "*.png"))
        self.data = [get_image(sample_file, self.config.image_size, is_crop=self.config.is_crop, resize_w=self.config.output_size, is_grayscale = (self.config.c_dim==1)) for sample_file in sample_files]
