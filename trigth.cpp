#include "trigth.h"

sem_t g_semTrig;

void trig_Handler(int sig)
{
    sem_post(&g_semTrig);
}

int initTrigInterrupt()
{
    int fd;
    int ret = 0;
    int oflags;
    do
    {
        if ((fd = open("/dev/camera_trigger", O_RDONLY)) < 0)
        {
            ret = -1;
            qDebug() << "open trig interrupt false...";
            break;
        }
        else
        {
            signal(SIGIO, trig_Handler); //triger signal posting
            fcntl(fd, F_SETOWN, getpid());
            oflags = fcntl(fd, F_GETFL);
            fcntl(fd, F_SETFL, oflags | FASYNC);
        }
    } while (0);
    return ret;
}

CTg::CTg(QObject *parent) : QThread(parent)
{
    qDebug() << "CTg started";
}

bool CTg::Init(int i)
{
    SYNCU_Init();
    m_handle = SYNCU_CreateHandle(0, 1, m_Trig_dev[i]);
    SYNCU_DisableTrig(m_handle);

    int ret = SYNCU_ReadVersion(m_handle, m_buf, sizeof(m_buf));
    if (ret < 0)
    {
        return false;
    }
    else
    {
        for (int i = 0; i < MAX_CHANNEL; i++)
        {
            SYNCU_SetTimeMode(m_handle, i, 30, 1000, 0);
        }
    }

    initTrigInterrupt();
    return true;
}

void CTg::run()
{
    Enable();
    qDebug() << "this is CTg" << QThread::currentThreadId();
    while (m_ret)
    {
        sem_wait(&g_semTrig); //triger signal consuming
        ReadTStamp();
        emit CamLaunch(); //sending reading signal to connected camer
    }
    Disable();
    m_ret = false;
}

void CTg::OpenRet()
{
    m_ret = true;
}

void CTg::CloseRet()
{
    m_ret = false;
}

bool CTg::Disable()
{
    qDebug()<<"SYNCU_Disabled";
    int ret =SYNCU_DisableTrig(m_handle);
    if(!ret)
    {
        return false;
    }

    return true;
}

bool CTg::Enable()
{
    int pt = SYNCU_EnableTrig(m_handle);
    if(pt<0)
    {
        return false;
    }
    return true;
}

int  CTg::ReadTStamp()
{
    TSyncuStamp tv;
    SYNCU_ReadTimeStamp(m_handle,&tv);
    if (SYNCU_ReadTimeStamp(m_handle, &tv) < 0)
    {
        printf("read timestamp failed\n");
    }
    else
    {
        printf("cpld:[%lld.%09lld]\n", tv.sec, tv.nan);
    }
    return 0;
}
