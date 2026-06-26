import napari
import numpy as np
import os
from skimage import io
from random import randint
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def create_folders_set(Set):
    """
    This function will create all the necessary folders that
    contain images for the training

    Parameters
    ----------
    Set : string
        Path of the created Set

    """

    try:
        os.mkdir(Set)
    except:
        pass

    try:
        os.mkdir(Set + "/currimg")
    except:
        pass

    try:
        os.mkdir(Set + "/previmg")
    except:
        pass

    try:
        os.mkdir(Set + "/nextimg")
    except:
        pass

    try:
        os.mkdir(Set + "/div_location")
    except:
        pass

class Annotator(object):
    def __init__(self, img_path, gt_folder, frame=1):
        self.img_path = img_path
        self.gt_folder = gt_folder
        self.current_frame = -1
        self.divisions_layers = []
        self.new_divisions_layer = None
        self.lock_view = False
        self.load_image()
        self.init_colors()
        self.create_viewer()
        current_step = self.viewer.dims.current_step
        new_step = (frame,) + current_step[1:]
        self.viewer.dims.current_step = new_step
        self.display_at_index(frame)
        create_folders_set(gt_folder)

        self.start()
        
    def init_colors(self):
        self.colors = []
        for i in range(100):
            self.colors.append("#%06X" % randint(16, 0xFFFFFF))
        logger.info("Colors created...")

    def load_image(self):
        self.img = io.imread(self.img_path)
        logger.info("Image loaded...")

    def create_viewer(self):
        self.viewer = napari.Viewer()
        self.viewer.add_image(self.img, name="image")
        
        self.viewer.layers.selection.events.changed.connect(self.on_layer_selected)
        self.viewer.dims.events.current_step.connect(self.on_current_step_changed)
                
        @self.viewer.bind_key("n")
        def next_layer(viewer):
            viewer.layers.select_next()
            for layer in viewer.layers.selection:
                if "centroid" in layer.name:
                    layer.mode = "select"

        @self.viewer.bind_key("m")
        def prev_layer(viewer):
            viewer.layers.select_previous()
            for layer in viewer.layers.selection:
                if "centroid" in layer.name:
                    layer.mode = "select"

        @self.viewer.bind_key("k")
        def delete_selection(viewer):
            # Get current selection
            current_selection = viewer.layers.selection.active
            
            # Set new selection
            viewer.layers.select_next()
            
            # Remove layer from layerlist
            viewer.layers.remove(current_selection)
            

    def start(self):
        napari.run()
        logger.info("Napari viewer started !")
        
    def on_layer_selected(self, event):
        if self.lock_view:
            return

        selection = self.viewer.layers.selection
        if len(selection) != 1:
            return
        selected_layer = selection.active
        if "centroid" not in selected_layer.name:
            return
        if len(selected_layer.data) != 2:
            return

        p1, p2 = selected_layer.data
        p1_x, p1_y = p1
        p2_x, p2_y = p2
        center = ((int(p1_x) + int(p2_x)) / 2, (int(p1_y) + int(p2_y)) / 2)
        w = max(300, abs(p2_x - p1_x))
        h = max(300, abs(p2_y - p1_y))
        ratio = max(self.img.shape[1] / h, self.img.shape[2] / w)
        self.viewer.camera.center = center
        self.viewer.camera.zoom = ratio 
               
    def on_current_step_changed(self, event):
        self.lock_view = True
        z = event.value[0]
        self.display_at_index(z)
        self.lock_view = False

    def display_at_index(self, index):
        if self.current_frame != -1:
            self.save_divisions()
            self.clear_divisions()
        self.current_frame = index
        self.divisions = self.load_divisions()
        print(self.divisions)
        self.draw_divisions()

    def load_divisions(self):
        divisions_path = os.path.join(self.gt_folder, "division_position" + str(self.current_frame) + ".npy")
        divisions = []
        if os.path.exists(divisions_path):
            divisions = list(np.load(divisions_path,allow_pickle=True))
        return divisions
        
    def draw_divisions(self, ):
        for k in range(0, len(self.divisions) - 1, 2):
            centroid = np.array(
                [
                    [self.divisions[k][0], self.divisions[k][1]],
                    [self.divisions[k + 1][0], self.divisions[k + 1][1]],
                ]
            )
            points_layer = self.viewer.add_points(centroid, size=5)
            points_layer.face_color = self.colors[k]
            self.divisions_layers.append(points_layer)

        # Add points
        self.new_divisions_layer = self.viewer.add_points(size=7)
        self.new_divisions_layer.mode = "add"
    
    def clear_divisions(self):
        for current_layer in self.divisions_layers:
            try:
                self.viewer.layers.remove(current_layer)
            except:
                pass
        if self.new_divisions_layer:
            self.viewer.layers.remove(self.new_divisions_layer)
    
        self.divisions_layers = []
        self.new_divisions_layer = None

    def save_divisions(self):
        division_position = self.gather_points()
        division_path = os.path.join(
            self.gt_folder, "division_position" + str(self.current_frame) + ".npy"
        )
        print(f"Saving #{len(division_position)} divisions to {division_path}")
        np.save(
            division_path,
            division_position,
        )

    def gather_points(self):
        division_position = []

        # Add points
        coordinates = np.round(self.new_divisions_layer.data).astype(int)

        for kkk in range(coordinates.shape[0]):
            division_position.append(
                np.array([coordinates[kkk, 0], coordinates[kkk, 1], self.current_frame])
            )

        for layer in self.viewer.layers:
            if "centroid" not in layer.name:
                continue
            new_data = np.round(layer.data).astype(int)
            for pt in new_data:
                division_position.append([pt[0], pt[1], self.current_frame])

        division_position = np.array(division_position)
        return division_position
