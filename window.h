#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>

class QPushButton;

class QLabel;

class QFileDialog;

class QSlider;

class GLWidget;

class QFrame;

class QTimer;

class QPointCloud;

class QTextEdit;

class Window : public QWidget
{
    Q_OBJECT

public:
    Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    QSlider *createSlider();

    GLWidget *glWidget;
    QSlider *xSlider;
    QSlider *ySlider;
    QSlider *zSlider;
    QPushButton *inputButton;
    QLabel *inputFile;
    QFileDialog *inputFileDialog;
    QPushButton *testButton;
    QLabel *testFile;
    QFileDialog *testFileDialog;
    QPushButton *startButton;
    QFrame *line;
    QTimer *timer;
    QPointCloud *input;
    QPointCloud *test;
    QTextEdit *console;

private slots:
    void updateInputLabel(QString filename);
    void updateTestLabel(QString filename);
    void enableStartButton();
};

#endif // WINDOW_H
