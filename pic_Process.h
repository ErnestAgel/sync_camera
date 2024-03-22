#ifndef CUDA_Init_H
#define CUDA_Init_H

#include "public.h"

class CCudapro : public QObject
{
    Q_OBJECT
public:
    CCudapro(int yuv_w, int yuv_h);
    void Init();
    void SetSignelpica(uint8_t *imgbuf);//recive and merge signel buf to forth buf
    void SetSignelpicb(uint8_t *imgbuf);
    void SetSignelpicc(uint8_t *imgbuf);
    void SetSignelpicd(uint8_t *imgbuf);
    void BuffReady();

protected:
    void ReadYUV(uint8_t *ResBuf, uint8_t *PreBuf, int resstart, int prestart, int resoffset, int preoffset, int size, int height);
    void Process_Image_Cuda(const uint8_t *src, int size);
    void Release();

signals:
    void I420_Comb();
    void Send_Rgb(uint8_t *rgb);

public:
    uint8_t *m_CombRgb;
    uint8_t *m_CombI420;
    int m_resWeight;
    int m_resHeight;
    int m_preWeight;
    int m_preHeight;

private:
    uint8_t *m_rgbCuda;
    uint8_t *m_yuvCuda;

    int m_yOffset;
    int m_uOffset;
    int m_vOffset;

    int m_preYsize;
    int m_preUsize;
    int m_preVsize;

    int m_resYsize;
    int m_resUsize;
    int m_resVsize;

    int m_preSize;
    int m_resSize;
    int m_count;
};
#endif // CUDA_Init_H
