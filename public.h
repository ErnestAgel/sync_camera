#pragma once
#ifndef PUBLICDEFINE_H
#define PUBLICDEFINE_H

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/prctl.h>

#include <poll.h>
#include <signal.h>
#include <queue>
#include <mqueue.h>
#include <semaphore.h>
#include <thread>
#include <poll.h>
#include <sched.h>
#include <pthread.h>
#include <malloc.h>

#include <assert.h>
#include <getopt.h> /* getopt_long() */
#include <fcntl.h> /* low-level i/o */
#include <errno.h>
#include <time.h>
#include <dlfcn.h>
#include <linux/videodev2.h>
#include <iostream>

#include "convert.h"
#include "convert_from.h"
#include "../scale.h"
#include "../format_conv.h"
#include "../basic_types.h"
#include <cuda_runtime.h>

#include <thread>
#include <QImage>
#include <QLabel>
#include <QMutex>
#include <QDebug>
#include <QObject>
#include <QString>
#include <QWidget>
#include <QPixmap>
#include <QVariant>
#include <QPainter>
#include <QTextEdit>
#include <QComboBox>
#include <QMetaType>
#include <QEventLoop>
#include <QReadLocker>
#include <QPushButton>
#include <QMessageBox>
#include <QApplication>
#include <QtConcurrent>
#include <QAbstractEventDispatcher>

#define ZERO 0
#define CAM_NUM 4
#define MAX_CHANNEL    8
#define BUFFER_LENGHT  4
#define CLEAR(x) memset(&(x), 0, sizeof(x))

#define TRIGCHECKBOX_NONE 0
#define TRIGCHECKBOX_TTY4 1 
#define TRIGCHECKBOX_TTY1 2 
#define TRIGCHECKBOX_TTY0 3 

#define CAMERCHECKBOX_NONE  0
#define CAMERCHECKBOX_720P  1
#define CAMERCHECKBOX_1080P 2

typedef struct{
      void *start;
      size_t length;
}TV4buffer;

typedef struct
{
    int width;
    int height;
    uint8_t *imgbuf;
    uint8_t *yuv[3];
} TPicInfo;

Q_DECLARE_METATYPE(TPicInfo)

#endif // PUBLICDEFINE_H
