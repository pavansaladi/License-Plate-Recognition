CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z','',' ','.']



num_classes = len(CHARS) +1

img_w, img_h = 256,64

num_train_images = 100000
num_validation_images = 10000

batch_size = 32
val_batch_size = 16

max_text_len = 13

