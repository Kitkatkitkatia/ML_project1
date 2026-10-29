import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap_scores(f1_grid,lambda_labels,gamma_labels):
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        f1_grid,
        xticklabels=lambda_labels,
        yticklabels=gamma_labels,
        annot=True,         
        fmt=".3f", 
        cmap="viridis", 
        cbar_kws={'label': 'Cross-Entropy Test Loss'}
    )

    plt.xlabel(r"$\lambda$ (Regularization Parameter)", fontsize=12)
    plt.ylabel(r"$\gamma$ (Learning Rate)", fontsize=12)
    plt.title(f"10-Fold CV F1 score\nBest: $\\gamma$={best_gamma:.1e}, $\\lambda$={best_lambda:.1e} (threshold={best_th:.3f})", fontsize=13)
    plt.tight_layout()
    plt.savefig('heatmap_f1_score.png')
    plt.show()


def cross_validation_visualization(lambds, rmse_tr, rmse_te, best_gamma):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color="b", label="train performance")
    plt.semilogx(lambds, rmse_te, marker=".", color="r", label="test performance")
    plt.xlabel("lambda")
    plt.ylabel("F1 score")
    # plt.xlim(1e-4, 1)
    plt.title(f"cross validation with fixed $\\gamma$={best_gamma:.1e}")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_f1_lambdas")
