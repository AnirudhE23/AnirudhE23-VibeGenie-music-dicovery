# Model evaluation metrics
# Implement evaluation functions, metrics calculation, and model assessment

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Handles model evaluation and assessment
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        pass
        
    def evaluate_reconstruction(self, model, X_test, y_test=None):
        """
        Evaluate autoencoder reconstruction quality
        
        Args:
            model: Trained autoencoder model
            X_test (np.array): Test data
            y_test (np.array): Target data (same as X_test for autoencoder)
            
        Returns:
            dict: Evaluation metrics
        """
        # For autoencoder, target is the same as input
        if y_test is None:
            y_test = X_test
            
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate reconstruction error per sample
        reconstruction_errors = np.mean((y_test - y_pred) ** 2, axis=1)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'mean_reconstruction_error': np.mean(reconstruction_errors),
            'std_reconstruction_error': np.std(reconstruction_errors)
        }
        
        return metrics, reconstruction_errors
        
    def evaluate_embeddings(self, embeddings, track_ids, df, n_samples=1000):
        """
        Evaluate embedding quality using similarity metrics
        
        Args:
            embeddings (np.array): Song embeddings
            track_ids (np.array): Track IDs
            df (pd.DataFrame): Original dataset
            n_samples (int): Number of samples to evaluate
            
        Returns:
            dict: Embedding evaluation metrics
        """
        # Sample random tracks for evaluation
        if n_samples > len(embeddings):
            n_samples = len(embeddings)
            
        sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[sample_indices]
        sample_track_ids = track_ids[sample_indices]
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(sample_embeddings)
        
        # Get track information
        sample_df = df[df['track_id'].isin(sample_track_ids)]
        
        # Calculate metrics
        metrics = {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'embedding_norm_mean': np.mean(np.linalg.norm(sample_embeddings, axis=1)),
            'embedding_norm_std': np.std(np.linalg.norm(sample_embeddings, axis=1))
        }
        
        return metrics, similarities, sample_df
        
    def plot_reconstruction_analysis(self, X_test, y_pred, save_path=None):
        """
        Plot reconstruction analysis
        
        Args:
            X_test (np.array): Original test data
            y_pred (np.array): Reconstructed data
            save_path (str): Path to save the plot
        """
        # Calculate reconstruction errors
        errors = X_test - y_pred
        error_magnitudes = np.linalg.norm(errors, axis=1)
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Reconstruction error distribution
        plt.subplot(1, 3, 1)
        plt.hist(error_magnitudes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Error Magnitude')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(error_magnitudes), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(error_magnitudes):.4f}')
        plt.legend()
        
        # Plot 2: Original vs Reconstructed (first feature)
        plt.subplot(1, 3, 2)
        plt.scatter(X_test[:, 0], y_pred[:, 0], alpha=0.5, s=1)
        plt.plot([X_test[:, 0].min(), X_test[:, 0].max()], 
                [X_test[:, 0].min(), X_test[:, 0].max()], 'r--', lw=2)
        plt.title('Original vs Reconstructed (Feature 1)')
        plt.xlabel('Original')
        plt.ylabel('Reconstructed')
        
        # Plot 3: Feature-wise reconstruction error
        plt.subplot(1, 3, 3)
        feature_errors = np.mean(errors ** 2, axis=0)
        plt.bar(range(len(feature_errors)), feature_errors)
        plt.title('Feature-wise Reconstruction Error')
        plt.xlabel('Feature Index')
        plt.ylabel('Mean Squared Error')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reconstruction analysis plot saved to {save_path}")
            
        plt.show()
        
    def plot_embedding_analysis(self, embeddings, save_path=None):
        """
        Plot embedding analysis using dimensionality reduction
        
        Args:
            embeddings (np.array): Song embeddings
            save_path (str): Path to save the plot
        """
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            
            plt.figure(figsize=(15, 5))
            
            # PCA analysis
            plt.subplot(1, 3, 1)
            pca = PCA(n_components=2)
            embeddings_pca = pca.fit_transform(embeddings)
            plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.6, s=1)
            plt.title(f'PCA of Embeddings\nExplained variance: {pca.explained_variance_ratio_.sum():.3f}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            
            # t-SNE analysis (on subset for speed)
            plt.subplot(1, 3, 2)
            if len(embeddings) > 5000:
                sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
                sample_embeddings = embeddings[sample_indices]
            else:
                sample_embeddings = embeddings
                
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_tsne = tsne.fit_transform(sample_embeddings)
            plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.6, s=1)
            plt.title('t-SNE of Embeddings')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            # Embedding norm distribution
            plt.subplot(1, 3, 3)
            embedding_norms = np.linalg.norm(embeddings, axis=1)
            plt.hist(embedding_norms, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('Embedding Norm Distribution')
            plt.xlabel('Embedding Norm')
            plt.ylabel('Frequency')
            plt.axvline(np.mean(embedding_norms), color='red', linestyle='--',
                       label=f'Mean: {np.mean(embedding_norms):.3f}')
            plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Embedding analysis plot saved to {save_path}")
                
            plt.show()
            
        except ImportError:
            print("scikit-learn not available for dimensionality reduction plots")
            
    def generate_evaluation_report(self, model, X_test, embeddings, track_ids, df):
        """
        Generate a comprehensive evaluation report
        
        Args:
            model: Trained model
            X_test (np.array): Test data
            embeddings (np.array): Song embeddings
            track_ids (np.array): Track IDs
            df (pd.DataFrame): Original dataset
            
        Returns:
            dict: Comprehensive evaluation report
        """
        print("Generating evaluation report...")
        
        # Evaluate reconstruction
        recon_metrics, recon_errors = self.evaluate_reconstruction(model, X_test)
        
        # Evaluate embeddings
        emb_metrics, similarities, sample_df = self.evaluate_embeddings(embeddings, track_ids, df)
        
        # Combine all metrics
        report = {
            'reconstruction_metrics': recon_metrics,
            'embedding_metrics': emb_metrics,
            'model_summary': {
                'total_embeddings': len(embeddings),
                'embedding_dimension': embeddings.shape[1],
                'test_samples': len(X_test)
            }
        }
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(f"Reconstruction Quality:")
        print(f"  - MSE: {recon_metrics['mse']:.6f}")
        print(f"  - MAE: {recon_metrics['mae']:.6f}")
        print(f"  - RMSE: {recon_metrics['rmse']:.6f}")
        print(f"\nEmbedding Quality:")
        print(f"  - Mean Similarity: {emb_metrics['mean_similarity']:.4f}")
        print(f"  - Embedding Norm Mean: {emb_metrics['embedding_norm_mean']:.4f}")
        print(f"  - Total Embeddings: {len(embeddings)}")
        print("="*50)
        
        return report
