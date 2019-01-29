//
// Created by Cameron Fleet on 28/01/2019.
//

#include <QPushButton>
#include <QHBoxLayout>
#include "ControlPanel.h"

ControlPanel::ControlPanel(QWidget *parent) : QWidget(parent) {

    QPushButton* playButton = new QPushButton("Play");
    QPushButton* displayButton = new QPushButton("Display");
    QPushButton* processButton = new QPushButton("Process");

    QHBoxLayout* mainLayout = new QHBoxLayout();
    mainLayout->addWidget(playButton);
    mainLayout->addWidget(displayButton);
    mainLayout->addWidget(processButton);

    QObject::connect( playButton, &QPushButton::clicked, [=](){system("pythonw ../../environment/main.py &");});
    this->setLayout(mainLayout);
}

