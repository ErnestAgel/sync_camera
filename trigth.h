#ifndef CTg_H
#define CTg_H

#include <QObject>
#include "public.h"
#include "syncu.h"
#include <QtConcurrent>
#include <QThread>

class CTg : public QThread
{
    Q_OBJECT
public:
    explicit CTg(QObject *parent = nullptr);
    bool Init(int i);    //init triger
    void OpenRet(); //change m_ret
    void CloseRet();
    void run(); //recoding QTthread run function

private:
    int  ReadTStamp();
    bool Enable();
    bool Disable();

signals:
    void CamLaunch(); //sending frame reading signal to connected camer

public:
    int m_tag;
    const char *m_Trig_dev[3]={"/dev/ttyTHS4","/dev/ttyTHS1","/dev/ttyTHS0"};

private:
    TSyncuStamp tv;
    void *m_handle;
    int  m_channum;
    char m_buf[64];
    volatile bool m_ret = false;
};

#endif // CTg_H
