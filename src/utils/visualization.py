import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def draw_emotion_bar(frame, probabilities, class_names, x=10, y=10, bar_width=120, bar_height=12):
    import cv2
    colors = {
        'angry':    (0,   0,   220),
        'disgust':  (0,   140, 0),
        'fear':     (180, 0,   180),
        'happy':    (0,   200, 200),
        'neutral':  (160, 160, 160),
        'sad':      (200, 100, 0),
        'surprise': (0,   180, 255),
    }
    top_idx = int(np.argmax(probabilities))

    for i, (cls, prob) in enumerate(zip(class_names, probabilities)):
        cy = y + i * (bar_height + 4)
        color = colors.get(cls, (255, 255, 255))
        filled_w = int(bar_width * prob)

        cv2.rectangle(frame, (x + 80, cy), (x + 80 + bar_width, cy + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x + 80, cy), (x + 80 + filled_w, cy + bar_height), color, -1)

        font_scale = 0.45
        thickness = 2 if i == top_idx else 1
        cv2.putText(frame, cls, (x, cy + bar_height - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(frame, f'{prob:.2f}', (x + 85 + bar_width, cy + bar_height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    return frame
