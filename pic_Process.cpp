#include "pic_Process.h"
#include <string.h>

/**************************************
 * 
 * init privte member relationship
 * 
 * *************************************/
CCudapro::CCudapro(int yuv_w,int yuv_h)
{
    m_count =0;
    m_yOffset = 0;
    m_uOffset = 0;
    m_vOffset = 0;

    m_preWeight =  yuv_w;
    m_preHeight =  yuv_h;

    m_preYsize = ((m_preWeight) * (m_preHeight));
    m_preUsize = ((m_preWeight / 2) * (m_preHeight / 2));
    m_preVsize = ((m_preWeight / 2) * (m_preHeight / 2));

    m_resWeight = yuv_w * 2;
    m_resHeight = yuv_h * 2;

    m_resYsize = ((m_resWeight) * (m_resHeight));
    m_resUsize = ((m_resWeight / 2) * (m_resHeight / 2));
    m_resVsize = ((m_resWeight / 2) * (m_resHeight / 2));

    m_preSize = ((m_preYsize) + (m_preUsize) + (m_preVsize));
    m_resSize = ((m_resYsize) + (m_resUsize) + (m_resVsize));

    m_CombI420 = NULL;
    m_CombRgb  = NULL;

    m_yuvCuda = NULL;
    m_rgbCuda  = NULL;
}

/**************************************
 * 
 * release all malloc buffer 
 * 
 * *************************************/
void CCudapro::Release()
{
    free(m_CombRgb);
    free(m_CombI420);
    cudaFree(m_yuvCuda);
    cudaFree(m_rgbCuda);
}

/**************************************
 * 
 * apply for all malloc buffer
 * 
 * *************************************/
void CCudapro::Init()
{
    int yuyv_image_size = m_resSize;
    int rgb_image_size  = m_resWeight * m_resHeight * 3;

    m_CombI420 = (uint8_t *)malloc(m_resSize);
    m_CombRgb  = (uint8_t *)malloc(m_resWeight * m_resHeight * 3);

    memset(m_CombI420, 0, m_resSize);
    memset(m_CombRgb, 0, m_resWeight * m_resHeight * 3);

    cudaError_t ret = cudaMalloc((void **)&m_yuvCuda, yuyv_image_size * sizeof(uint8_t));
    if (cudaSuccess != ret)
    {
        printf("Fail to allocate cuda memory %d\n", ret);
        exit(EXIT_FAILURE);
    }
    ret = cudaMemset((void *)m_yuvCuda, 0, yuyv_image_size * sizeof(uint8_t));
    if (cudaSuccess != ret)
    {
        printf("Fail to set cuda memory1 %d\n", ret);
        exit(EXIT_FAILURE);
    }
    ret = cudaMalloc((void **)&m_rgbCuda, rgb_image_size * sizeof(uint8_t));
    if (cudaSuccess != ret)
    {
        printf("Fail to allocate cuda memory %d\n", ret);
        exit(EXIT_FAILURE);
    }
    ret = cudaMemset((void *)m_rgbCuda, 0, rgb_image_size * sizeof(uint8_t));
    if (cudaSuccess != ret)
    {
        printf("Fail to set cuda2 memory %d\n", ret);
        exit(EXIT_FAILURE);
    }
}

/**************************************
 * 
 * combine each signel buf to m_CombI420
 * 
 * top left 
 * 
 * *************************************/
void CCudapro::SetSignelpica(uint8_t *imgbuf)
{
    if(NULL != imgbuf)
    {
        m_count++;
        qDebug()<<"m_stPicSignel[0] get imgbuf"<<m_count;
        do
        {
            m_yOffset = 0;
            m_uOffset = m_resYsize;
            m_vOffset = m_resYsize + m_resUsize;
            qDebug()<<"------------process_yuv_combin CASE0-----------";
            ReadYUV(m_CombI420, imgbuf, m_yOffset, 0, m_resWeight, m_preWeight, m_preWeight, m_preHeight);
            ReadYUV(m_CombI420, imgbuf, m_uOffset, m_preYsize, m_resWeight / 2, m_preWeight / 2, m_preWeight / 2, m_preHeight / 2);
            ReadYUV(m_CombI420, imgbuf, m_vOffset, m_preYsize + m_preUsize, m_resWeight / 2, m_preWeight / 2, m_preWeight / 2, m_preHeight / 2);
            qDebug()<<"------------process_yuv_combin CASE0 finished-----------";
        }while(0);
        if(CAM_NUM == m_count)
        {
            emit I420_Comb();
        }
    }
    else
    {
        qDebug()<<"---------m_stPicSignel[0] get imgbuf failed----------";
    }
    return;
}

/**************************************
 * 
 * combine each signel buf to m_CombI420
 * 
 * top right
 * 
 * *************************************/
void CCudapro::SetSignelpicb(uint8_t *imgbuf)
{
    if(NULL != imgbuf)
    {
        m_count++;
        qDebug()<<"m_stPicSignel[1] get imgbuf"<<m_count;
        do
        {
            m_yOffset = m_preWeight;
            m_uOffset = m_resYsize + m_preWeight / 2;
            m_vOffset = m_resYsize + m_resUsize + m_preWeight / 2;
            qDebug()<<"------------process_yuv_combin CASE1-----------";
            ReadYUV(m_CombI420, imgbuf, m_yOffset, 0, m_resWeight, m_preWeight, m_preWeight, m_preHeight);
            ReadYUV(m_CombI420, imgbuf, m_uOffset, m_preYsize, m_resWeight / 2, m_preWeight / 2, m_preWeight / 2, m_preHeight / 2);
            ReadYUV(m_CombI420, imgbuf, m_vOffset, m_preYsize + m_preUsize, m_resWeight / 2, m_preWeight / 2, m_preWeight / 2, m_preHeight / 2);
            qDebug()<<"------------process_yuv_combin CASE1 finished-----------";
        }while(0);
        if(CAM_NUM == m_count)
        {
            emit I420_Comb();
        }
    }
    else
    {
        qDebug()<<"---------m_stPicSignel[1] get imgbuf failed----------";
    }
    return;
}

/**************************************
 * 
 * combine each signel buf to m_CombI420
 * 
 * buttom left 
 * 
 * *************************************/
void CCudapro::SetSignelpicc(uint8_t *imgbuf)
{
    if(NULL != imgbuf)
    {
        m_count++;
        qDebug()<<"m_stPicSignel[2] get imgbuf"<<m_count;
        do
        {
            m_yOffset = m_preYsize * 2;
            m_uOffset = m_resYsize + m_preYsize / 2;
            m_vOffset = m_resYsize + m_resUsize + m_preYsize / 2;
            qDebug()<<"------------process_yuv_combin CASE2-----------";
            ReadYUV(m_CombI420, imgbuf, m_yOffset, 0, m_resWeight, m_preWeight, m_preWeight, m_preHeight);
            ReadYUV(m_CombI420, imgbuf, m_uOffset, m_preYsize, m_resWeight / 2, m_preWeight / 2, m_preWeight / 2, m_preHeight / 2);
            ReadYUV(m_CombI420, imgbuf, m_vOffset, m_preYsize + m_preUsize, m_resWeight / 2, m_preWeight / 2, m_preWeight / 2, m_preHeight / 2);
            qDebug()<<"------------process_yuv_combin CASE2 finished-----------";
        }while(0);
        if(CAM_NUM == m_count)
        {
            emit I420_Comb();
        }
    }
    else
    {
        qDebug()<<"---------m_stPicSignel[2] get imgbuf failed----------";
    }
    return;
}

/**************************************
 * 
 * combine each signel buf to m_CombI420
 * 
 * buttom right 
 * 
 * *************************************/
void CCudapro::SetSignelpicd(uint8_t *imgbuf)
{
    if(NULL !=imgbuf)
    {
        m_count++;
        qDebug()<<"m_stPicSignel[3] get imgbuf"<<m_count;
        do
        {
            m_yOffset = m_preWeight + m_preWeight * m_preHeight * 2;
            m_uOffset = m_resYsize + m_preYsize / 2 + m_preWeight / 2;
            m_vOffset = m_resYsize + m_resUsize + m_preYsize / 2 + m_preWeight / 2;
            qDebug()<<"------------process_yuv_combin CASE3-----------";
            ReadYUV(m_CombI420, imgbuf, m_yOffset, 0, m_resWeight, m_preWeight, m_preWeight, m_preHeight);
            ReadYUV(m_CombI420, imgbuf, m_uOffset, m_preYsize, m_resWeight / 2, m_preWeight / 2, m_preWeight / 2, m_preHeight / 2);
            ReadYUV(m_CombI420, imgbuf, m_vOffset, m_preYsize + m_preUsize, m_resWeight / 2, m_preWeight / 2, m_preWeight / 2, m_preHeight / 2);
            qDebug()<<"------------process_yuv_combin CASE3 finished-----------";
        }while(0);
        if(CAM_NUM == m_count)
        {
            emit I420_Comb();
        }
    }
    else
    {
        qDebug()<<"---------m_stPicSignel[3] get imgbuf failed----------";
    }
    return;
}

/**************************************
 * 
 * check recived buf number 
 * 
 * notiyf cuda yuv->rgb
 * 
 * *************************************/
void CCudapro::BuffReady()
{
    if(CAM_NUM != m_count)
    {
        qDebug()<<"imbuf not ready";
    }
    else
    {
        m_count = 0;
        Process_Image_Cuda(m_CombI420,m_resSize);
        fflush(stdout);

    }
}

/**************************************
 * 
 * yuv buf copy function
 * 
 * *************************************/
void CCudapro::ReadYUV(uint8_t *ResBuf,uint8_t*PreBuf, int resstart, int prestart, int resoffset, int preoffset, int size, int height)
{
    for (int k = 0; k < height; k++)
    {
        memcpy(ResBuf + resstart + k * (resoffset), PreBuf + prestart + k * (preoffset), size);
    }
}

/**************************************
 * 
 * process yuv->rgb
 * 
 * *************************************/
void CCudapro::Process_Image_Cuda(const uint8_t *src, int size)
{
    int yuv_size = size * sizeof(uint8_t);

    cudaError_t ret = cudaMemcpy(m_yuvCuda, src, yuv_size, cudaMemcpyHostToDevice);//CPU yuv -> GPU

    if (cudaSuccess != ret)
    {
        printf("cudaMemcpy fail %d\n", ret);
    }

    CUDA_yu12_to_rgb(m_yuvCuda, m_rgbCuda, m_resWeight, m_resHeight);//GPU: yuv -> rgb

    int rgb_size = size * 2 * sizeof(uint8_t);

    ret = cudaMemcpy(m_CombRgb, m_rgbCuda, rgb_size, cudaMemcpyDeviceToHost);//GPU RGB -> CPU 

    if (cudaSuccess != ret)
    {
        printf("cudaMemcpy fail %d\n", ret);
    }

    emit Send_Rgb(m_CombRgb);   //sending rgb to QLabel
}

