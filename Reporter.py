from path import Path
import csv

class Reporter():
    def __init__(self, args) -> None:
        self.args = args

        save_path = Path(self.args.name)
        self.args.save_path = '/content/drive/MyDrive/VinAI/Motion segmentation/checkpoints_sfmlearner'/save_path #/timestamp
        print('=> will save everything to {}'.format(self.args.save_path))
        self.args.save_path.makedirs_p()

        with open(self.args.save_path/self.args.log_summary, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['train_loss', 'validation_loss'])

    def update_log_summary(self, train_loss, decisive_error):
        with open(self.args.save_path/self.args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    def create_report(self):
        pass