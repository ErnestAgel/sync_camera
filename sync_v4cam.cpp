#include "sync_v4cam.h"
#include <QThread>

CCamer::CCamer()
{
    qDebug() << "camera newed";
}

CCamer::~CCamer()
{
    CapStop();
    free(m_stRawI420.imgbuf);
    free(m_stSmI420.imgbuf);
    Close();
    qDebug() << " ~CCamer finished";
}

/**************************************
 * 
 * init Camer width && height 
 * 
 * init TPicInfo member size && logic
 * 
 * ***********************************/
void CCamer::Init(int i, int in_width, int in_height)
{
    m_dev = m_video[i];

    CLEAR(m_stRawI420);
    CLEAR(m_stSmI420);
    m_stRawI420.width = in_width;
    m_stRawI420.height = in_height;

    m_stRawI420.yuv[0] = NULL;
    m_stRawI420.yuv[1] = NULL;
    m_stRawI420.yuv[2] = NULL;
    m_stRawI420.imgbuf = NULL;

    m_stSmI420.width = in_width / 2;
    m_stSmI420.height = in_height / 2;

    m_stSmI420.yuv[0] = NULL;
    m_stSmI420.yuv[1] = NULL;
    m_stSmI420.yuv[2] = NULL;
    m_stSmI420.imgbuf = NULL;

    m_stRawI420.imgbuf = (uint8_t *)malloc(m_stRawI420.width * m_stRawI420.height * 3 / 2);
    m_stSmI420.imgbuf =  (uint8_t *)malloc(m_stSmI420.width * m_stSmI420.height * 3 / 2);

    m_stSmI420.yuv[0] = m_stSmI420.imgbuf;
    m_stSmI420.yuv[1] = m_stSmI420.imgbuf + m_stSmI420.width * m_stSmI420.height;
    m_stSmI420.yuv[2] = m_stSmI420.yuv[1] + m_stSmI420.width * m_stSmI420.height / 4;

    m_stRawI420.yuv[0] = m_stRawI420.imgbuf;
    m_stRawI420.yuv[1] = m_stRawI420.imgbuf + m_stRawI420.width * m_stRawI420.height;
    m_stRawI420.yuv[2] = m_stRawI420.yuv[1] + m_stRawI420.width * m_stRawI420.height / 4;

    memset(m_stRawI420.imgbuf, 0, m_stRawI420.width * m_stRawI420.height * 3 / 2);
    memset(m_stSmI420.imgbuf, 0, m_stSmI420.width * m_stSmI420.height * 3 / 2);
}

/**************************************
 * 
 * camer open && set && call mmapinit
 * 
 * ***********************************/
bool CCamer::CamSet()
{
    struct v4l2_capability cap;
    struct v4l2_format fmt;

    m_dvideo = open(m_dev, O_RDWR, 0);
    qDebug() << "this is m_dev" << m_dev;
    if (-1 == m_dvideo)
    {
        qDebug() << "Cannot open:" << m_dev << " " << errno << "" << strerror(errno);
        return false;
    }
    qDebug() << "the number of m_dvideo" << m_dvideo;
    //_________________open_camer__________
    if (-1 == xioctl(m_dvideo, VIDIOC_QUERYCAP, &cap))
    {
        if (EINVAL == errno)
        {
            qDebug() << "is no V4L2 device\n"<< m_dev;
            return false;
        }
        else
        {
            qDebug() << "open camer failed";
            return false;
        }
    }
    qDebug() << "---------------------------67------------------------";
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
    {
        qDebug() << "%s is no video capture device\n"
                 << m_dev;
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING))
    {
        qDebug() << "%s does not support streaming i/o\n"<< m_dev;
        return false;
    }
    CLEAR(fmt);
    //_________________set_camer__________
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = m_stRawI420.width;
    fmt.fmt.pix.height = m_stRawI420.height;
    fmt.fmt.pix.pixelformat = m_v4l2_format;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

    if (-1 == xioctl(m_dvideo, VIDIOC_S_FMT, &fmt))
    {
        errno_exit("VIDIOC_S_FMT");
    }

    qDebug() << "cam Inited";
    MmapInit();
    return true;
}

/**************************************
 * 
 * camer  buffer mmaping
 * 
 * ***********************************/
void CCamer::MmapInit()
{
    qDebug() << "CCamer------------96";
    struct v4l2_requestbuffers req;
    CLEAR(req);
    req.count = BUFFER_LENGHT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(m_dvideo, VIDIOC_REQBUFS, &req))
    {
        if (EINVAL == errno)
        {
            qDebug() << "%s does not support memory mapping\n"<< m_dev;
            exit(EXIT_FAILURE);
        }
        else
        {
            qDebug() << "CCamer------------110";
            errno_exit("VIDIOC_REQBUFS");
        }
    }

    if (req.count < 2)
    {
        qDebug() << "Insufficient buffer memory on %s\n"<< m_dev;
        exit(EXIT_FAILURE);
    }

    buffers = (TV4buffer *)calloc(req.count, sizeof(*buffers));
    qDebug() << "buffer calloc";
    if (!buffers)
    {
        qDebug() << "Out of memory\n";
        exit(EXIT_FAILURE);
    }

    for (m_nbuffers = 0; m_nbuffers < req.count; ++m_nbuffers)
    {
        struct v4l2_buffer buf;
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = m_nbuffers;

        if (-1 == xioctl(m_dvideo, VIDIOC_QUERYBUF, &buf))
        {
            errno_exit("VIDIOC_QUERYBUF");
        }

        buffers[m_nbuffers].length = buf.length;
        buffers[m_nbuffers].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, m_dvideo, buf.m.offset);
        qDebug() << "CCamer------------139";
        if (MAP_FAILED == buffers[m_nbuffers].start)
        {
            errno_exit("mmap");
        }
    }
    qDebug() << "Capturing------------180";
    Capturing();
}

/**************************************
 * 
 * camer  capturing 
 * 
 * ***********************************/
void CCamer::Capturing()
{
    for (unsigned int i = 0; i < BUFFER_LENGHT; ++i)
    {
        struct v4l2_buffer buf;
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (-1 == xioctl(m_dvideo, VIDIOC_QBUF, &buf))
        {
            qDebug() << "CCamer------------153";
            errno_exit("VIDIOC_QBUF");
        }
    }
    qDebug() << "Capturing finished---199";
    return;
}

/**************************************
 * 
 * declear YUYV->I420
 * 
 * ***********************************/
void CCamer::YUYTOI420(uint8_t *src_yuy2, int src_stride_yuy2, TPicInfo *out)
{
    libyuv::YUY2ToI420(src_yuy2, src_stride_yuy2 * 2,
                       out->yuv[0], out->width,
                       out->yuv[1], (out->width / 2),
                       out->yuv[2], (out->width / 2),
                       out->width, out->height);
}

/**************************************
 * 
 * scale I420 / 4
 * 
 * ***********************************/
void CCamer::I420Zoome(TPicInfo *b_I420, TPicInfo *m_I420)
{
    libyuv::I420Scale(b_I420->yuv[0],
                      b_I420->width,
                      b_I420->yuv[1],
                      b_I420->width / 2,
                      b_I420->yuv[2],
                      b_I420->width / 2,
                      b_I420->width,
                      b_I420->height,

                      m_I420->yuv[0],
                      m_I420->width,
                      m_I420->yuv[1],
                      m_I420->width / 2,
                      m_I420->yuv[2],
                      m_I420->width / 2,
                      m_I420->width,
                      m_I420->height,
                      libyuv::kFilterNone);
}

/**************************************
 * 
 * read sigel camer frame
 * 
 * ***********************************/
void CCamer::ReadFrame()
{
    struct v4l2_buffer buf;
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(m_dvideo, VIDIOC_DQBUF, &buf)) //pull out buf list
    {
        errno_exit("VIDIOC_DQBUF");
    }

    YUYTOI420((uint8_t *)buffers[buf.index].start, m_stRawI420.width, &m_stRawI420);
    I420Zoome(&m_stRawI420, &m_stSmI420);

    emit SendImg(m_stSmI420.imgbuf);    // sending scaled yuv to pic_process
    qDebug() << "the img sending:" << QThread::currentThreadId();

    if (-1 == xioctl(m_dvideo, VIDIOC_QBUF, &buf)) //push buf list back
    {
        errno_exit("VIDIOC_QBUF");
    }

    return;
}

/**************************************
 * 
 * Turn on camer capture
 * 
 * ***********************************/
bool CCamer::CapStart()
{
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(m_dvideo, VIDIOC_STREAMON, &type))
    {
        return false;
        errno_exit("VIDIOC_STREAMOFF");
    }
    return true;
}

/**************************************
 * 
 * Turn off camer capture
 * 
 * ***********************************/
bool CCamer::CapStop()
{
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(m_dvideo, VIDIOC_STREAMOFF, &type))
    {
        errno_exit("VIDIOC_STREAMOFF");
    }
    return true;
}

/**************************************
 * 
 * close camer device
 * 
 * ***********************************/
void CCamer::Close()
{
    qDebug() << "CapStoped";
    for (unsigned int i = 0; i < BUFFER_LENGHT; i++)
    {
        if (-1 == munmap(buffers[i].start, buffers[i].length))
        {
            errno_exit("munmap");
        }
    }

    free(buffers);

    if (-1 == close(m_dvideo))
    {
        errno_exit("Close m_dvideo");
    }

    return;
}


void CCamer::errno_exit(const char *s)
{
    qDebug() <<QObject:: tr("s:") << s <<QObject::  tr("errno:") << errno <<QObject::  tr("strerror:") << strerror(errno);
    exit(EXIT_FAILURE);
}

int CCamer::xioctl(int fh, int request, void *arg)
{
    int r;
    do
    {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}
