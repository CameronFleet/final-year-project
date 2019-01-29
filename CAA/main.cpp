#include <iostream>
#include <QApplication>
#include <QDebug>
#include <QPushButton>
#include <QVBoxLayout>
#include "ControlPanel.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    auto controlPanel = new ControlPanel();

    auto mainLayout = new QVBoxLayout();
    mainLayout->addWidget(controlPanel);

    QWidget* widget = new QWidget();
    widget->setLayout(mainLayout);

    widget->show();
    return app.exec();
}