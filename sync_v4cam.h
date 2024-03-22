#ifndef SYNC_CV4H
#define SYNC_CV4H

#include "public.h"

class CCamer : public QObject
{
    Q_OBJECT
    TV4buffer *buffers;

public:
    CCamer();
    ~CCamer();
    void Init(int i, int in_width, int in_height);
    bool CamSet(); //include MmapInit
    bool CapStart();
    bool CapStop();
    void Capturing();
    void ReadFrame();
    void Close();

private:
    void MmapInit();
    void YUYTOI420(uint8_t *src_yuy2, int src_stride_yuy2, TPicInfo *out);
    void I420Zoome(TPicInfo *b_I420, TPicInfo *m_I420);
    void errno_exit(const char *s);
    int  xioctl(int fh, int request, void *arg);


signals:
    void SendImg(uint8_t *);//sending scaled i420 buf to pic_process

private:
    int m_dvideo;
    TPicInfo m_stSmI420;
    TPicInfo m_stRawI420;
    unsigned int m_nbuffers;
    const char *m_dev = NULL;
    int m_v4l2_format = V4L2_PIX_FMT_YUYV;
    const char *m_video[8] = {"/dev/video0", "/dev/video2", "/dev/video3", "/dev/video4",
                              "/dev/video5", "/dev/video1", "/dev/video6", "/dev/video7"};
};
#endif // SYNC_CV4H
