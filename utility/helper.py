import os
import re
import matplotlib.pyplot as plt
import numpy as np


def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def record(path, loss_loger, rec_loger, ndcg_loger):
    fig = plt.figure(figsize=(15, 5))
    epochs = 50 * np.arange(len(rec_loger)) + 50
    ax = fig.add_subplot(1, 3, 1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.plot(epochs, loss_loger)
    ax = fig.add_subplot(1, 3, 2)
    ax.set_xlabel('epoch')
    ax.set_ylabel('recall')
    for i in range(5):
        ax.plot(epochs, rec_loger[:, i])
    ax.legend(['1', '2', '3', '4', '5'])
    ax = fig.add_subplot(1, 3, 3)
    ax.set_xlabel('epoch')
    ax.set_ylabel('NDCG')
    for i in range(5):
        ax.plot(epochs, ndcg_loger[:, i])
    ax.legend(['1', '2', '3', '4', '5'])
    plt.show()
    plt.savefig(path + '/result.png')
    file = open(path + '/result.txt', 'w+')
    title = 'epoch\tloss\t'
    for i in range(1, 6):
        title = title + 'recall{}\t'.format(i)
    for i in range(1, 6):
        title = title + 'NDCG{}\t'.format(i)
    title = title + '\n'
    file.write(title)
    for epoch, loss, rec, ndcg in zip(epochs, loss_loger, rec_loger, ndcg_loger):
        line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(epoch, loss, rec[0], rec[1], rec[2], rec[3],
                                                                         rec[4], ndcg[0], ndcg[1], ndcg[2], ndcg[3],
                                                                         ndcg[4])
        file.write(line)
    file.close()
