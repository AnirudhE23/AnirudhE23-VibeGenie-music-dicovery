# Training loop and logic
# Implement your training loops, callbacks, and training utilities

import tensorflow as tf
import matplotlib.pyplot as plt
from .config import MODEL_CONFIG, CALLBACKS_CONFIG
import numpy as np


class ModelTrainer:
    """
    Handles the training process for the music recommendation model
    """
    
    def __init__(self, model, config=None):
        """
        Initialize the trainer
        
        Args:
            model: The model to train (autoencoder)
            config (dict): Training configuration
        """
        self.model = model
        self.config = config or MODEL_CONFIG
        self.history = None
        
    def compile_model(self):
        """Compile the model with optimizer, loss, and metrics"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        print("Model compiled successfully")
        
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=CALLBACKS_CONFIG['early_stopping']['patience'],
                restore_best_weights=CALLBACKS_CONFIG['early_stopping']['restore_best_weights'],
                monitor=CALLBACKS_CONFIG['early_stopping']['monitor']
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=CALLBACKS_CONFIG['reduce_lr']['factor'],
                patience=CALLBACKS_CONFIG['reduce_lr']['patience'],
                min_lr=CALLBACKS_CONFIG['reduce_lr']['min_lr'],
                monitor=CALLBACKS_CONFIG['reduce_lr']['monitor']
            ),
            tf.keras.callbacks.ModelCheckpoint(
                CALLBACKS_CONFIG['model_checkpoint']['filepath'],
                save_best_only=CALLBACKS_CONFIG['model_checkpoint']['save_best_only'],
                monitor=CALLBACKS_CONFIG['model_checkpoint']['monitor']
            )
        ]
        return callbacks
        
    def train(self, X_train, X_val, verbose=1):
        """
        Train the model
        
        Args:
            X_train (np.array): Training data
            X_val (np.array): Validation data
            verbose (int): Verbosity level
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        print("Starting model training...")
        
        # Compile model if not already compiled
        if not hasattr(self.model, 'optimizer'):
            self.compile_model()
            
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            X_train, X_train,  # Autoencoder learns to reconstruct input
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("Training completed!")
        return self.history
        
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
        
    def get_training_summary(self):
        """
        Get a summary of the training results
        
        Returns:
            dict: Training summary
        """
        if self.history is None:
            return None
            
        # Get best metrics
        best_epoch = np.argmin(self.history.history['val_loss']) + 1
        best_val_loss = min(self.history.history['val_loss'])
        best_val_mae = min(self.history.history['val_mae'])
        
        return {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_mae': best_val_mae,
            'final_val_loss': self.history.history['val_loss'][-1],
            'final_val_mae': self.history.history['val_mae'][-1],
            'total_epochs': len(self.history.history['loss'])
        }
