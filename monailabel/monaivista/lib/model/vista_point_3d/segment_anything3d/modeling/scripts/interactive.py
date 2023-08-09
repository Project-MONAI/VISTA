import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, simpledialog
import nibabel as nib
import copy
import fire
import pdb
from .infer import InferClass
configs='configs_total'
inferer = InferClass(
    config_file=[f'{configs}/hyper_parameters.yaml',f'{configs}/network.yaml',f'{configs}/transforms_train.yaml',f'{configs}/transforms_infer.yaml',f'{configs}/transforms_validate.yaml']
)
class samm_visualizer():
    def __init__(self):
        self.clicked_points = []
        self.data = None
        self.mask_plot = None
        self.mask = None
        self.data_path = None
        self.circle_artists = []

    def select_data_file(self):
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select Data File", initialfile=self.data_path)
        if not file_path:
            print("No file selected.")
            exit()
        # Load data from NIfTI file
        try:
            nifti_img = nib.load(file_path)
            data = nifti_img.get_fdata()
            if len(data.shape) == 4:
                data = data[..., 0]  # Extract last element along the 4th dimension
        except FileNotFoundError:
            print("File not found.")
            exit()
        except nib.filebasedimages.ImageFileError:
            print("Invalid NIfTI file.")
            exit()
        self.data = data
        self.data_path = file_path

    def generate_mask_dummy(self):
        print('generated mask')
        self.mask = np.zeros_like(self.data, dtype=bool) * np.nan

        for x, y, z, point_label in self.clicked_points:
            # mask[data == scalar] = True
            self.mask[y-10:y+10,x-10:x+10,z] = 1
        
    def generate_mask(self):
        point = []
        point_label = []
        for p in self.clicked_points:
            point.append([p[1],p[0],p[2]])
            point_label.append(p[3])
        if len(point) == 0:
            point = None
            point_label = None
        else:
            point = np.array(point)[np.newaxis,...]
            point_label = np.array(point_label)[np.newaxis,...]
        if self.class_label is None:
            label_prompt = None
        else:
            label_prompt = np.array([int(self.class_label)])[np.newaxis,...]
        mask = inferer.infer({'image':self.data_path}, point, point_label, label_prompt, save_mask=True)[0]
        mask = mask.data.cpu().numpy() > 0.5
        mask = mask.astype(np.float32)
        mask[mask==0] = np.nan
        self.mask = mask

    def display_3d_slices(self):
        fig, ax = plt.subplots()
        assert self.data is not None, 'Load data first.'
        ax.volume = self.data
        ax.index = self.data.shape[2] // 2
        ax.imshow(self.data[:, :, ax.index], cmap='gray')
        ax.set_title(f"Slice {ax.index}")
        if self.mask is not None:
            self.mask_plot = ax.imshow(np.zeros_like(self.mask[:, :, ax.index]) * np.nan, cmap='Reds', alpha=0.5)
        self.update_slice(ax)
        fig.canvas.mpl_connect('scroll_event', self.process_scroll)
        fig.canvas.mpl_connect('button_press_event', self.process_click)
        plt.show()

    def process_class(self):
        self.class_label = simpledialog.askfloat("Scalar Input", "Enter scalar value:")

    def process_scroll(self, event):
        ax = event.inaxes
        if event.button == 'up':
            self.previous_slice(ax)
        elif event.button == 'down':
            self.next_slice(ax)

    def previous_slice(self, ax):
        if ax is None:
            return
        ax.index = (ax.index - 1) % ax.volume.shape[2]
        self.update_slice(ax)

    def next_slice(self,ax):
        if ax is None:
            return
        ax.index = (ax.index + 1) % ax.volume.shape[2]
        self.update_slice(ax)

    def update_slice(self,ax):
        # remove circles 
        while(len(self.circle_artists)>0):
            ca = self.circle_artists.pop()
            ca.remove()
        # plot circles
        for x, y, z, label in self.clicked_points:
            if z == ax.index:
                color = 'red' if label == 1 else 'blue'
                circle_artist = plt.Circle((x, y), 1, color=color, fill=False)
                self.circle_artists.append(circle_artist)
                ax.add_artist(circle_artist)
        ax.images[0].set_array(ax.volume[:, :, ax.index])
        if self.mask is not None and self.mask_plot is not None:
            self.mask_plot.set_data(self.mask[:, :, ax.index])
            self.mask_plot.set_visible(True)
        ax.set_title(f"Slice {ax.index}")
        ax.figure.canvas.draw()

    def process_click(self, event):
        ax = event.inaxes

        if ax is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            z = ax.index
            print(f"Clicked coordinates: x={x}, y={y}, z={z}")
            if event.key == "control":
                point_label = 0
            else:
                point_label = 1
            self.clicked_points.append((x, y, z, point_label))
            self.update_slice(ax)
    
    def run(self):
        # File selection
        user_input = '0'
        inferer.clear_cache()
        while True:
            if user_input == '':
                break
            if user_input == '0':
                self.select_data_file()
                inferer.clear_cache()
            if user_input=='0' or user_input=='1':
                inferer.clear_cache()
                # clear points
                self.clicked_points = []
                self.mask = None
            self.display_3d_slices()
            if user_input=='0' or user_input=='2':
                self.process_class()
                self.mask = None
            # Generate mask based on user inputs
            self.generate_mask()
            # Overlay mask with the image
            self.display_3d_slices()
            user_input = input('Press enter to exist, press 0 to restart, press 1 to reselect points, press 2 to reselect class, press 3 to add point')

if __name__ == "__main__":
    from monai.utils import optional_import
    fire, _ = optional_import("fire")
    # using python -m interactive run
    fire.Fire(samm_visualizer)