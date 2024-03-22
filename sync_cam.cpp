#include "ui_sync_cam.h"
#include "sync_cam.h"

CSyncCam::CSyncCam(QWidget *parent) : QWidget(parent),
                                      ui(new Ui::CSyncCam)
{
    ui->setupUi(this);
    m_trigThread = new CTg;
    m_label = new QLabel;
    m_label->setFrameStyle(QFrame::Box | QFrame::Raised);
    m_label->setGeometry(270, 20, 1280, 720);
    m_label->setParent(this);
}

CSyncCam::~CSyncCam()
{
    delete ui;
}

/*********************************************
 * creat pic_porcess obj && thread
 *
 * creat cam obj && thread
 *
 * connect cam soucre to pic_process
 *********************************************/
bool CSyncCam::CamInit(int width, int height)
{
    do
    {
        m_sub[4] = new QThread;
        pic = new CCudapro(width / 2, height / 2);
        pic->moveToThread(m_sub[4]);
        pic->Init();
        pic->BuffReady();
        m_sub[4]->start();
        for (int i = ZERO; i < CAM_NUM; i++)
        {
            m_cam[i] = new CCamer;
            m_sub[i] = new QThread;
            m_cam[i]->moveToThread(m_sub[i]);
            m_sub[i]->start();
            m_cam[i]->Init(i, width, height);
            if (!m_cam[i]->CamSet())
            {
                goto bad;
            }
            m_cam[i]->CapStart();
            connect(m_trigThread, &CTg::CamLaunch, m_cam[i], &CCamer::ReadFrame, Qt::QueuedConnection); //connect emit launch to each cam readframe
        }
        connect(m_cam[0], &CCamer::SendImg, pic, &CCudapro::SetSignelpica, Qt::QueuedConnection); //connect cam0 to top left corner
        connect(m_cam[1], &CCamer::SendImg, pic, &CCudapro::SetSignelpicb, Qt::QueuedConnection); //connect cam1 to top right corner
        connect(m_cam[2], &CCamer::SendImg, pic, &CCudapro::SetSignelpicc, Qt::QueuedConnection); //connect cam2 to bottom left corner
        connect(m_cam[3], &CCamer::SendImg, pic, &CCudapro::SetSignelpicd, Qt::QueuedConnection); //connect cam3 to bottom right corner

        connect(pic, &CCudapro::I420_Comb, pic, &CCudapro::BuffReady, Qt::QueuedConnection);  //sending signal to process I420->RGB88
        connect(pic, &CCudapro::Send_Rgb, this, &CSyncCam::RgbDisplay, Qt::QueuedConnection); //sending ImgRGB to show label
        goto nice;
    } while (0);

nice:
    QMessageBox::information(this, "Camer setting", "setting Camer success", QMessageBox::Ok);
    return true;
bad:
    QMessageBox::information(this, "Camer setting", "Camer info not be set correctly,Please reset Camer", QMessageBox::Ok);
    return false;
}

/*********************************************
 *
 * init triger
 *
 *********************************************/
bool CSyncCam::TrigInit()
{
    if (m_trigThread->Init(m_trigThread->m_tag))
    {
       return true;
    }
    else
    {
        return false;
    }
}

/*********************************************
 * 
 * creat RGB Display slot && label display info
 * 
**********************************************/
void CSyncCam::RgbDisplay(uint8_t *rgb)
{
    qDebug() << "------------RgbDisplay-----------";
    m_image = new QImage(rgb, m_width, m_height, QImage::Format_RGB888);
    QPixmap pix = QPixmap::fromImage(*m_image, Qt::AutoColor);
    m_label->setPixmap(pix);
}

/*********************************************
 * 
 * define Start pushButton logic
 * 
**********************************************/
void CSyncCam::on_ui_Start_clicked()
{
    if (ZERO == ui->ui_Bbox->currentIndex() || ZERO == ui->ui_CamBox->currentIndex() ||
        ZERO == m_width || ZERO == m_height)
    {
        QMessageBox::warning(this, "Application Warning", "Triger or Camer not be initialized, Please initialize first!",QMessageBox::Ok);
        return;
    }
    else if(!m_trigThread->isRunning())
    {
        m_trigThread->start();
        m_trigThread->OpenRet();
    }
    else
    {
        m_trigThread->OpenRet();
        qDebug() << "this is ui thread" << QThread::currentThreadId();
    }
}

/*********************************************
 * 
 * define Stop pushButton logic
 * 
**********************************************/
void CSyncCam::on_ui_Stop_clicked()
{
    m_trigThread->CloseRet();
    QMessageBox::information(this, "Camer Status", "Camer stoped", QMessageBox::Ok);
}

/*********************************************
 * 
 * define Triger check box logic
 * 
**********************************************/
void CSyncCam::on_ui_Bbox_currentIndexChanged(int index)
{
    switch (index)
    {
    case TRIGCHECKBOX_NONE:
        break;

    case TRIGCHECKBOX_TTY4:
        m_trigThread->m_tag = TRIGCHECKBOX_TTY4 - 1;
        TrigInit();
        break;

    case TRIGCHECKBOX_TTY1:
        m_trigThread->m_tag = TRIGCHECKBOX_TTY1 - 1;
        TrigInit();
        break;

    case TRIGCHECKBOX_TTY0:
        m_trigThread->m_tag = TRIGCHECKBOX_TTY0 - 1;
        TrigInit();
        break;

    default:
        break;
    }
}

/*********************************************
 * 
 * define Camer check box logic
 * 
**********************************************/
void CSyncCam::on_ui_CamBox_currentIndexChanged(int index)
{
    switch (index)
    {
    case CAMERCHECKBOX_NONE:
        m_width = 0;
        m_height = 0;
        QMessageBox::information(this, "Camer setting", "Camer info not be set", QMessageBox::Ok);
        break;

    case CAMERCHECKBOX_720P:
        m_width = 1280;
        m_height = 720;
        CamInit(m_width, m_height);
        break;

    case CAMERCHECKBOX_1080P:
        CamInit(m_width, m_height);
        m_width = 1920;
        m_height = 1080;
        break;

    default:
        QMessageBox::information(this, "Camer setting", "Camer info not be set", QMessageBox::Ok);
        return;
        break;
    }
}
