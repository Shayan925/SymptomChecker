import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from kivy.config import Config
from kivy.uix.layout import Layout
Config.set('graphics', 'width', '700')
Config.set('graphics', 'height', '800')

from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivy.metrics import dp
import pandas as pd
import numpy as np
import tensorflow as tf

# Read in the datasets
df = pd.read_csv('data/dataset.csv')
df_severities = pd.read_csv('data/Symptom-severity.csv')
df_precautions = pd.read_csv('data/symptom_precaution.csv')
df_descriptions = pd.read_csv('data/symptom_Description.csv')

# List of diseases and symptoms
diseases = list(df['Disease'].unique())
symptoms = sorted(list(df_severities['Symptom'].unique()))


class FirstScreen(Screen): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.selected = []
        self.cnt = 0

        # The main layout for the first screen
        main_layout = BoxLayout(orientation="vertical", size=(self.width, self.height),
                                padding=[dp(5), dp(5), 0, dp(50)])

        # Scroll view for seeing all the symptoms
        scroll_symptoms = ScrollView(size_hint=(1, 1), 
                                    do_scroll_y=True, do_scroll_x=False,
                                    bar_width=dp(20), scroll_type=['bars', 'content'],
                                    scroll_wheel_distance=dp(40))

        # Stack layout to hold all the toggle buttons
        hold_buttons = StackLayout(size_hint_y=None)
        hold_buttons.bind(minimum_height=hold_buttons.setter('height'))

        # Create all the toggle buttons
        for i in range(len(symptoms)):
            t_btn = ToggleButton(text=f"{symptoms[i]}", size_hint=(0.33, None), height=dp(50),
                                background_color=[0, 0.3, 1, 0.75])
            t_btn.bind(on_press=self.button_toggled)
            hold_buttons.add_widget(t_btn)

        # Add to screen
        scroll_symptoms.add_widget(hold_buttons)
        main_layout.add_widget(MDLabel(text="Select All Your Symptoms", halign="center", size_hint_y=None, height=dp(40), font_style="H4"))
        main_layout.add_widget(MDLabel(text="*Do not select more than 17 boxes", halign="center", size_hint_y=None, height=dp(30), theme_text_color="Hint"))
        main_layout.add_widget(scroll_symptoms)
        self.add_widget(main_layout)

    def button_toggled(self, togglebutton):
        if togglebutton.state == "down":
            self.selected.append(togglebutton.text)
            self.cnt += 1
        else:
            self.selected.remove(togglebutton.text)
            self.cnt -= 1
        
        if self.cnt > 17:
            self.show_dialog(self.cnt)

    def show_results(self):
        data = self.selected.copy()

        # Replace symptoms with unique id
        for i in range(len(self.selected)):
            data[i] = df_severities[df_severities["Symptom"] == self.selected[i]].index.values.astype(int)[0] + 1
        
        if 17-len(data) < 0:
            return 0

        for _ in range(17-len(data)):
            data.append(0)
        
        data = np.asarray(data, dtype=np.int64)
        data = data.reshape((-1, 17))

        SecondScreen.display_text(data)
    
    
    # Create and show the pop up dialog
    def show_dialog(self, num):
        ok_btn = MDFlatButton(text="OK", on_release=self.close_dialog)
        self.dialog = MDDialog(title="Warning", 
                                text=f"{num} boxes are selected.\nPlease only select 17 or less boxes", 
                                buttons=[ok_btn])
        self.dialog.open()
    
    # Close the pop up dialog
    def close_dialog(self, obj):
        self.dialog.dismiss()


class SecondScreen(Screen): 
    # Scroll view for seeing all the text
    scroll_text = ScrollView(size_hint=(1, 1), 
                            do_scroll_y=True, do_scroll_x=False,
                            bar_width=dp(20), scroll_type=['bars', 'content'])

    # Stack layout to hold all the toggle buttons
    hold_text = BoxLayout(orientation="vertical", size_hint_y=None, padding=[dp(15)])
    hold_text.bind(minimum_height=hold_text.setter('height'))

    def __init__(self, **kw):
        super().__init__(**kw)

        self.scroll_text.add_widget(self.hold_text)
        self.add_widget(self.scroll_text)
        

    def display_text(data):

        # Load model
        tf.keras.backend.clear_session()
        new_model = tf.keras.models.load_model('symptom_checker.model')

        # Make prediction
        predictions = new_model.predict(data)
        
        label_1 = MDLabel(text="Top Results", font_style="H2", size_hint=(1, None))
        SecondScreen.hold_text.add_widget(label_1)

        for i in range(1, 4):
            illness = diseases[np.argsort(np.max(predictions, axis=0))[-i]]
            label_2 = MDLabel(text=f"{i}. {illness}", font_style="H4", size_hint=(1, None))
            SecondScreen.hold_text.add_widget(label_2)
            label_3 = MDLabel(text="Description", font_style="H5", size_hint=(1, None))
            SecondScreen.hold_text.add_widget(label_3)
            label_4 = MDLabel(text=f"{df_descriptions.loc[df_descriptions['Disease'] == illness]['Description'].values[0]}", font_style="H6", size_hint=(1, None))
            SecondScreen.hold_text.add_widget(label_4)
            label_5 = MDLabel(text="Precautions", font_style="H5", size_hint=(1, None))
            SecondScreen.hold_text.add_widget(label_5)

            for j in range(1, 5):
                try:
                    label_6 = MDLabel(text=f"- {df_precautions.loc[df_precautions['Disease'] == illness][f'Precaution_{j}'].values[0].title()}", font_style="H6", size_hint=(1, None), height=dp(25))
                    SecondScreen.hold_text.add_widget(label_6)
                except Exception as e:
                    pass
        
    
    def clear(self):
        self.hold_text.clear_widgets()
            

class WindowManager(ScreenManager):
    pass


class CheckSymptomsApp(MDApp):
    def build(self):
        self.title = "Check Symptoms"
        kv = Builder.load_file("source.kv")
        return kv


if __name__ == "__main__":
    CheckSymptomsApp().run()


    