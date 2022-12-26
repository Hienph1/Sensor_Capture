import threading
import logging
import time

def thread_func(name):
    logging.info('Thread %s: starting', name)
    time.sleep(2)
    logging.info('Thread %s: finished', name)

def calculator():
    print(1)

if __name__ == '__main__':
    format = '%(asctime)s: %(message)s'
    logging.basicConfig(format = format, level = logging.INFO, datefmt = '%H:%M:%S')

    # Chay 1 thread
    # logging.info('Main  : before creating thread')
    # x = threading.Thread(target = thread_func, args = (1,))
    # logging.info('Main  : before running thread')
    # x.start()
    # logging.info('Main  : wait for the thread to finish')
    # x.join()
    # logging.info('Main  : Done !!!')

    # Chay nhieu thread
    threads = list()
    for i in range(3):
        logging.info('Main  : create and start thread %d', i)
        x = threading.Thread(target = thread_func, args = (i,))
        threads.append(x)
        x.start()

    for i, thread in enumerate(threads):
        logging.info('Main  : before joining thread %d', i)
        thread.join()
        logging.info('Main  : thread %d done', i)
    
    logging.info('Main  : Done !!!')


    # Su dung ThreadPoolExecutor
    # import concurrent.futures as concurrent
    # with concurrent.ThreadPoolExecutor(max_workers=3) as executor:
    #     executor.map(thread_func, range(2))