#include <QtGui>

#include "glwidget.h"
#include "window.h"
#include <QTimer>
#include <QResource>

Window::Window()
{
    this->input = new QPointCloud();
    this->input->loadDataFromPCD(":/bun000.pcd");
    this->test = new QPointCloud();
    this->test->loadDataFromPCD(":/bun315.pcd");
    rotate_point_cloud(this->test->data, 0, 0.7, 0);

    glWidget = new GLWidget(this->input, this->test);

    xSlider = createSlider();
    ySlider = createSlider();
    zSlider = createSlider();

    inputButton = new QPushButton("Input");
    inputFile = new QLabel("No Input");
    testButton = new QPushButton("Test");
    testFile = new QLabel("No Test");
    inputFileDialog = new QFileDialog;
    testFileDialog = new QFileDialog;

    startButton = new QPushButton("Start");
    startButton->setEnabled(false);

    timer = new QTimer();

    console  = new QTextEdit();
    glWidget->setConsole(console);

    connect(xSlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setXRotation(int)));
    connect(glWidget, SIGNAL(xRotationChanged(int)), xSlider, SLOT(setValue(int)));
    connect(ySlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setYRotation(int)));
    connect(glWidget, SIGNAL(yRotationChanged(int)), ySlider, SLOT(setValue(int)));
    connect(zSlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setZRotation(int)));
    connect(glWidget, SIGNAL(zRotationChanged(int)), zSlider, SLOT(setValue(int)));
    connect(inputButton, SIGNAL(clicked()), inputFileDialog, SLOT(open()));
    connect(inputFileDialog, SIGNAL(fileSelected(QString)), this, SLOT(updateInputLabel(QString)));
    connect(inputFileDialog, SIGNAL(fileSelected(QString)), glWidget, SLOT(loadInputCloudFromPCD(QString)));
    connect(testButton, SIGNAL(clicked()), testFileDialog, SLOT(open()));
    connect(testFileDialog, SIGNAL(fileSelected(QString)), this, SLOT(updateTestLabel(QString)));
    connect(testFileDialog, SIGNAL(fileSelected(QString)), glWidget, SLOT(loadTestCloudFromPCD(QString)));
    connect(startButton, SIGNAL(clicked()), glWidget, SLOT(doICP()));
    connect(timer, SIGNAL(timeout()), this, SLOT(enableStartButton()));

    timer->start(100);

    line = new QFrame();
    line->setObjectName(QString::fromUtf8("line"));
    line->setGeometry(QRect(60, 110, 118, 3));
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    QWidget *main = new QWidget;
    QHBoxLayout *mainLayout2 = new QHBoxLayout;
    mainLayout2->addWidget(glWidget);

    QWidget *rightBar = new QWidget();
    QVBoxLayout *rightBarLayout = new QVBoxLayout();
    rightBar->setLayout(rightBarLayout);
    QWidget *bunnies = new QWidget;
    QVBoxLayout *bunniesLayout = new QVBoxLayout;
    bunniesLayout->addWidget(inputFile);
    bunniesLayout->addWidget(inputButton);
    bunniesLayout->addWidget(testFile);
    bunniesLayout->addWidget(testButton);
    bunniesLayout->addWidget(line);
    bunniesLayout->addWidget(startButton);
    bunnies->setLayout(bunniesLayout);
    rightBarLayout->addWidget(bunnies);
    QWidget *sliders = new QWidget;
    QHBoxLayout *slidersLayout = new QHBoxLayout;
    slidersLayout->addWidget(xSlider);
    slidersLayout->addWidget(ySlider);
    slidersLayout->addWidget(zSlider);
    sliders->setLayout(slidersLayout);
    rightBarLayout->addWidget(sliders);
    mainLayout2->addWidget(rightBar);
    main->setLayout(mainLayout2);
    mainLayout->addWidget(main);
    mainLayout->addWidget(console);
    setLayout(mainLayout);

    xSlider->setValue(15 * 16);
    ySlider->setValue(345 * 16);
    zSlider->setValue(0 * 16);
    setWindowTitle(tr("Bunnies...do ICP!!"));
}

QSlider *Window::createSlider()
{
    QSlider *slider = new QSlider(Qt::Vertical);
    slider->setRange(0, 360 * 16);
    slider->setSingleStep(16);
    slider->setPageStep(15 * 16);
    slider->setTickInterval(15 * 16);
    slider->setTickPosition(QSlider::TicksRight);
    return slider;
}

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

void Window::updateInputLabel(QString filename)
{
    QRegExp rx;
    rx.setPattern(".*\\/(.*)\\.[^.]+$");
    rx.setPatternSyntax(QRegExp::RegExp2);
    rx.indexIn(filename);
    filename = rx.capturedTexts().last();
    inputFile->setText(filename);
}

void Window::updateTestLabel(QString filename)
{
    QRegExp rx;
    rx.setPattern(".*\\/(.*)\\.[^.]+$");
    rx.setPatternSyntax(QRegExp::RegExp2);
    rx.indexIn(filename);
    filename = rx.capturedTexts().last();
    testFile->setText(filename);
}

void Window::enableStartButton()
{
    if(input->data && test->data)
        startButton->setEnabled(true);
}
