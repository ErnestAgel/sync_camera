#ifndef CSyncCam_H
#define CSyncCam_H

#include "public.h"
#include "sync_v4cam.h"
#include "pic_Process.h"
#include "trigth.h"

namespace Ui {
class CSyncCam;
}

class CSyncCam : public QWidget
{
    Q_OBJECT
public:
    explicit CSyncCam(QWidget *parent = 0);
    ~CSyncCam();
    bool TrigInit();
    bool CamInit(int width,int height);
    void RgbDisplay(uint8_t *rgb);

public:
    int m_width;
    int m_height;
    Ui::CSyncCam *ui;
    CCudapro *pic;

private slots:
    void on_ui_Start_clicked();
    void on_ui_Stop_clicked();
    void on_ui_Bbox_currentIndexChanged(int index);
    void on_ui_CamBox_currentIndexChanged(int index);

private:
    CCamer *m_cam[4];
    QImage *m_image;
    QLabel *m_label;
    QThread *m_sub[5];
    CTg *m_trigThread;

};
#endif // CSyncCam_H
