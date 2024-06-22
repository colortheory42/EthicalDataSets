import pretrain_model
import interface

if __name__ == "__main__":
    print("Starting pre-training...")
    pretrain_model.main()
    print("Pre-training completed. Starting user feedback loop...")
    interface.user_feedback_loop()
