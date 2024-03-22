#include <QApplication>
#include "sync_cam.h"
#include "public.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CSyncCam w;

    w.show();
    return a.exec();
}
