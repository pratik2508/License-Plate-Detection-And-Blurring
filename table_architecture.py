from _datetime import datetime
import psycopg2
import os

import detect_batch2
import logging
from logging.handlers import TimedRotatingFileHandler
from logging import Formatter
import traceback
import threading


handler = TimedRotatingFileHandler('log-traffic', when='MIDNIGHT', backupCount=7)
formatter = Formatter(fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
logger = logging.getLogger('werkzeug')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


class Wrap:
    def __init__(self):
        try:
            self.conn=psycopg2.connect(host="10.1.1.168", database="License_Plate", user="postgres", password="postgres")
            # cur = (self.conn).cursor()  # connection for License Plate db

            self.conn1=psycopg2.connect(host="10.1.1.168", database="ai_ml_hub", user="postgres", password="postgres")
            # connection for ai-ml_hub db

        except Exception as e:
            logger.error(msg='Connection not established ' + str(e)
                             + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

    def batch_call(self):
        cur = (self.conn).cursor()
        cur1 = (self.conn1).cursor()

        cur1.execute("SELECT Proc_Count FROM catalog.GPU_Server_info WHERE Server_Name='Server_3'")
        s = cur1.fetchone()[0]
        print('Gpu_Process count:-',+ s)
        if s < 2:
            try:
                cur1.execute(
                    "select folder_path,State from catalog.process_queue where model_name='LP' and state !=4 "
                    "order by datetime")
                fetch = cur1.fetchone()
                input_path = fetch[0]

            except Exception as e:
                print("No task of License Plate is in queue table")
                logger.error(
                    msg='No task of License Plate is in queue table :-\t' + str(e) + '\tTraceback\t' + '~'
                    .join(str(traceback.format_exc()).split('\n')))

            # input_path1 parameter is for ubuntu
            # input_path parameter  is for windows path
            input_path1 = input_path.replace('%20', ' ').replace('\\', '/').replace('//', '/').replace(
                '/10.1.1.', '/mnt/vol1/10.1.1.')
            print(input_path1)

            if os.path.exists(input_path1):
                route = input_path.split('\\')[-6]+'_'+input_path.split('\\')[-3]+'_'+input_path.split('\\')[-1]
                route = route.lower()
                print(route)

                pro_count, route1 = self.batch_process(input_path, route, input_path1)

                task = detect_batch2.detect_batch1.delay(input_path, route1, pro_count)
                print(task)

                cur.execute("insert into catalog.task_details (Folder_path, Task_id) values('{input_path}','{task}')"
                            .format(input_path=input_path,task=task))
                (self.conn).commit()
                print('task_id inserted')

                cur1.execute(
                    "DELETE FROM catalog.process_queue WHERE model_name='LP' and folder_path='{path}'"
                        .format(path=input_path))
                self.conn.commit()
                self.conn1.commit()


            else:
                cur1.execute(
                    "update catalog.Process_Queue set state=4,  remarks='Folder_Path does not exist'"
                    " where folder_path='{Folder_path}'".format(Folder_path=input_path))
                print('folder_path does not exist ')
                self.conn.commit()
                self.conn1.commit()

        else:
            print("Process can't be executed as process count is 2 ")

        threading.Timer(300, self.batch_call).start()

    def batch_process(self, input_path, route, input_path1 ):
        print('secondary_table_entered')
        cur = (self.conn).cursor()
        cur1=(self.conn1).cursor()

        cur.execute(
            "Select route,state from catalog.lp_folder_path_info where folder_path='{path}' and state != 1 "
            "order by sno desc" .format(path=input_path))
        # matching input_path from input_path of function batch_call()

        fetch = cur.fetchone()   # route, state value for folder_path_info table

        cur1.execute(
            "Select state from catalog.process_queue where model_name='LP' and folder_path='{path}' "
                .format(path=input_path))
        fetch1 = cur1.fetchone()[0]  # state value of process queue table

        processed_count = 0

        if fetch != None:
            route1 = fetch[0]
            print(route1)

            if fetch[1] == 2 and fetch1 == 2:  # interrupted start from that count
                print('interrupted start from that count')
                cur.execute("Select processed_Count,route from catalog.lp_folder_path_info where folder_path='{path}'"
                            " and state =2"
                            .format(path=input_path))
                processed_count = cur.fetchone()[0]
                print(processed_count)

            elif fetch[1] == 2 and fetch1 == 3:  # interrupted, reprocess

                try:
                    print('Interrupted, reprocess')

                    cur1.execute(
                        "Select state from  catalog.process_queue where model_name='LP' and folder_path='{path}'"
                                .format(path=input_path))
                    var = cur1.fetchone()
                    print(var)

                    cur.execute(
                        "Select req_stat from catalog.lp_folder_path_info where folder_path='{path}' order by Date desc"
                            .format(path=input_path))

                    stat = cur.fetchone()[0]

                    stat1 = stat + 1
                    route1 = str(route) + '_' + str(stat1)

                    cur.execute(
                        "insert into catalog.lp_folder_path_info (folder_path, date,state, processed_count,total_count,"
                        "route,req_stat)"
                        "VALUES('{input_path}','{date}',1, 0,'{total_count}','{route}',{req_stat})"
                            .format(input_path=input_path, date=datetime.now(), total_count=len(os.listdir(input_path1)),
                                    route=route1, req_stat=stat + 1))
                    print(route1)

                    (self.conn).commit()

                    cur.execute("CREATE TABLE public.{route} (Image_name text,Box text,flag text)"
                                .format(route=route1, stat=stat))

                except Exception as e:
                    logger.error(msg='Exception occurred\t' + str(e) + '\tTraceback\t' + '~'
                                 .join(str(traceback.format_exc()).split('\n')))

            elif fetch[1] == 3 and fetch1 == 3:   # to implement resume condition if process table state is 2
                cur1.execute("Select state from  catalog.process_queue where model_name='lp' and folder_path='{path}'"
                            .format(path=input_path))
                var = cur1.fetchone()
                print("Only reprocess")

                cur.execute(
                    "Select req_stat from catalog.lp_folder_path_info where folder_path='{path}' order by Date desc"
                            .format(path=input_path))

                stat = cur.fetchone()[0]

                stat1 = stat + 1
                route1 = str(route) + '_' + str(stat1)
                print(route1)

                cur.execute(
                    "insert into catalog.lp_folder_path_info (folder_path, date,state, processed_count,total_count,"
                    "route,req_stat)"
                    "VALUES('{input_path}','{date}',1, 0,'{total_count}','{route}',{req_stat})"
                        .format(input_path=input_path, date=datetime.now(), total_count=len(os.listdir(input_path1)),
                                route=route1, req_stat=stat+1))
                print(route1)

                cur.execute("CREATE TABLE public.{route} (Image_name text,Box text,flag text)"
                            .format(route=route1, stat=stat))

            self.conn.commit()
            self.conn1.commit()

        if fetch == None:
            # route1 = input_path1.split('/')[-1] + '_1'
            route1=input_path.split('\\')[-6] + '_' + input_path.split('\\')[-3] + '_' + input_path.split('\\')[-1]+'_1'
            route1=route1.lower()

            print(route1)

            print('new_process_case')
            cur.execute(
                "insert into catalog.lp_folder_path_info (folder_path, date,state, processed_count,total_count,route,req_stat)"
                "VALUES('{input_path}','{date}',1, 0,'{total_count}','{route}',1)"
                    .format(input_path=input_path, date=datetime.now(),
                            total_count=len(os.listdir(input_path1)), route=route1))
            self.conn.commit()

            cur.execute("CREATE TABLE public.{route} (Image_name text, Box text ,flag text)"
                            .format(route=route1))

            self.conn.commit()

        # wrapper use
        cur1.execute("UPDATE catalog.gpu_server_info SET proc_count=proc_Count + 1 WHERE server_name='Server_3' ;""")
        (self.conn1).commit()
        return processed_count, route1


obj = Wrap()
obj.batch_call()
