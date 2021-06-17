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
            writer.writerow(['train_loss', 'validation_loss', 'abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])

    def update_log_summary(self, train_loss, decisive_error, errors):
        with open(self.args.save_path/self.args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            results = [train_loss, decisive_error] + errors
            writer.writerow(results)
            
    def create_report(self):
        # if self.args.train:
        #     report_path = self.args.save_path/'train_report.txt'
        #     with open(report_path, 'a') as f:
        pass
