import os
import time
import random
from datetime import datetime
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from scipy.signal import savgol_filter
from scipy.signal.windows import hann
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from audiolazy import lazy_lpc as LPC
from matplotlib import pyplot as plt


ROOT_PATH = os.path.expanduser(os.path.join('~', 'sandbox', 'surat'))
DEVICE = torch.device('cuda')
SHAPES_COUNT = 51


class Data(Dataset):
    def __init__(self, transforms=None, shiftRandom=True, preview=False, validationAudioPath=None):
        self.transforms = transforms
        self.shiftRandom = shiftRandom and not preview and validationAudioPath is None
        self.preview = preview
        self.count = None
        if validationAudioPath is not None and not os.path.exists(validationAudioPath):
            raise

        animFPS = 24
        audioSampleRate = 16000

        if validationAudioPath is not None:
            temp, audioSamples = wavfile.read(
                os.path.join(
                    ROOT_PATH,
                    'data',
                    validationAudioPath
                )
            )
        else:
            temp, audioSamples = wavfile.read(
                os.path.join(
                    ROOT_PATH,
                    'data',
                    'inputSpeech.wav'
                )
            )
        self.count = int((audioSamples.shape[0] / temp) * animFPS)

        autoCorrelationsPath = os.path.join(
            ROOT_PATH,
            'data',
            'precalculatedAutoCorrelations.npy'
        )
        if validationAudioPath is not None:
            autoCorrelationsPath = os.path.join(
                ROOT_PATH,
                'data',
                'precalculatedAutoCorrelations_{}.npy'.format(os.path.basename(validationAudioPath.split('.')[0]))
            )
        precalculate = not os.path.exists(autoCorrelationsPath)
        if precalculate:
            print('Precalculating autocorrelations to {}'.format(autoCorrelationsPath))
            self.audioCorrelationsPerFrame = np.array([])

            # in case audio doesn't have the desired sample rate
            if temp != audioSampleRate:
                audioSamples = resample(
                    audioSamples,
                    int(audioSampleRate * audioSamples.shape[0] / temp)
                )

            # TODO remove DC component
            audioSamples -= np.mean(audioSamples)

            # take left? channel as mono if stereo
            if audioSamples.shape[-1] == 2:
                audioSamples = audioSamples[:, 0]

            audioFrameIndexRange = range(0, audioSamples.shape[0], int(audioSampleRate / 1000))
            totalCount = len(audioFrameIndexRange)
            for p, i in enumerate(audioFrameIndexRange):
                startTime = time.time()

                audioFrame = np.append(
                    np.roll(audioSamples, (i * -1) + 128)[:128],
                    np.roll(audioSamples, (i * -1))[:128]
                )
                #  apply hann window
                hannWindow = hann(256)
                audioFrame = hannWindow * audioFrame
                # autocorrelation
                filt = LPC.lpc['autocorrelation'](audioFrame, 32)
                coefficients = np.array(filt.numerator[1:])
                self.audioCorrelationsPerFrame = np.append(self.audioCorrelationsPerFrame, coefficients)

                print('{}/{} - {:03f}'.format(p + 1, totalCount, time.time() - startTime))

            self.audioCorrelationsPerFrame = self.audioCorrelationsPerFrame.reshape(-1, 32)
            np.save(
                autoCorrelationsPath,
                self.audioCorrelationsPerFrame.reshape(-1, 32)
            )
        else:
            print('Loading precalculated auto-correlations from {}'.format(autoCorrelationsPath))
            self.audioCorrelationsPerFrame = np.load(autoCorrelationsPath)

        self.audioFramesCount = self.audioCorrelationsPerFrame.shape[0]

        if self.shiftRandom:
            self.audioCorrelationsPerFrame = torch.from_numpy(self.audioCorrelationsPerFrame).float().to(DEVICE)
            self.halfShift = int(audioSampleRate / 1000) * 8  # 8ms shift left or right
        else:
            audioFrameInputsPath = os.path.join(
                ROOT_PATH,
                'data',
                'precalculatedAudioFrameInputs.npy'
            )
            if validationAudioPath is not None:
                audioFrameInputsPath = os.path.join(
                    ROOT_PATH,
                    'data',
                    'precalculatedAudioFrameInputs_{}.npy'.format(os.path.basename(validationAudioPath.split('.')[0]))
                )
            precalculate = not os.path.exists(audioFrameInputsPath)
            if precalculate:
                print('Precalculating audio frame inputs to {}'.format(audioFrameInputsPath))
                self.inputValues = np.array([])
                for i in range(self.count):
                    audioIdxRoll = int((i / self.count) * self.audioFramesCount)
                    inputValue = np.concatenate(
                        (
                            np.roll(self.audioCorrelationsPerFrame, (audioIdxRoll * -1) + 32, axis=0)[:32],
                            np.roll(self.audioCorrelationsPerFrame, (audioIdxRoll * -1), axis=0)[:32]
                        ),
                        axis=None
                    ).reshape(1, 64, 32)
                    self.inputValues = np.append(
                        self.inputValues,
                        inputValue
                    )
                np.save(
                    audioFrameInputsPath,
                    self.inputValues.reshape(-1, 1, 64, 32)
                )
            else:
                print('Loading precalculated audio frame inputs from {}'.format(autoCorrelationsPath))
                self.inputValues = np.load(audioFrameInputsPath)
            self.inputValues = torch.from_numpy(self.inputValues).float().to(DEVICE)

        if validationAudioPath is not None:
            self.targetValues = np.zeros((self.count, SHAPES_COUNT))
        else:
            self.targetValues = np.load(
                os.path.join(
                    ROOT_PATH,
                    'data',
                    'shapeValuesPerFrame.npy'
                )
            )
            self.count = self.targetValues.shape[0]

        self.targetValues = torch.from_numpy(
            self.targetValues.reshape(-1, SHAPES_COUNT)
        ).float().view(-1, SHAPES_COUNT).to(DEVICE)

    def __getitem__(self, i):
        if i < 0:  # for negative indexing
            i = self.count + i

        if self.preview:
            return (
                torch.Tensor([i]).long().to(DEVICE),
                self.inputValues[i] / 100.,
                (self.targetValues[i] - .5) * 2.
            )

        if self.shiftRandom:
            randomShift = random.randint(-1 * self.halfShift, self.halfShift)
            randomShiftPair = random.randint(-1 * self.halfShift, self.halfShift)
            audioIdxRoll = int((i / self.count) * self.audioFramesCount + randomShift)
            audioIdxRollPair = int(((i + 1) / self.count) * self.audioFramesCount + randomShiftPair)
            inputValue = torch.cat(
                (
                    torch.cat(
                        (
                            torch.roll(self.audioCorrelationsPerFrame, (audioIdxRoll * -1) + 32, dims=0)[:32],
                            torch.roll(self.audioCorrelationsPerFrame, (audioIdxRoll * -1), dims=0)[:32]
                        ),
                        dim=0
                    ),
                    torch.cat(
                        (
                            torch.roll(self.audioCorrelationsPerFrame, (audioIdxRollPair * -1) + 32, dims=0)[:32],
                            torch.roll(self.audioCorrelationsPerFrame, (audioIdxRollPair * -1), dims=0)[:32]
                        ),
                        dim=0
                    )
                ),
                dim=0
            ).view(2, 1, 64, 32).float().to(DEVICE)
        else:
            inputValue = self.inputValues[i:i + 2]

        targetValue = self.targetValues[i:i + 2]

        return (
            torch.Tensor([i]).long().to(DEVICE),
            inputValue / 100.,
            (targetValue - .5) * 2.
        )

    def __len__(self):
        if self.preview:
            return self.count
        return self.count - 1  # for pairs


class Model(nn.Module):
    def __init__(self, moodSize, filterMood=False):
        super(Model, self).__init__()

        self.shapesLen = SHAPES_COUNT

        self.formantAnalysis = nn.Sequential(
            nn.Conv2d(1, 72, (1, 3), (1, 2), (0, 1), 1),
            nn.BatchNorm2d(72),
            nn.LeakyReLU(),
            nn.Conv2d(72, 108, (1, 3), (1, 2), (0, 1), 1),
            nn.BatchNorm2d(108),
            nn.LeakyReLU(),
            nn.Conv2d(108, 162, (1, 3), (1, 2), (0, 1), 1),
            nn.BatchNorm2d(162),
            nn.LeakyReLU(),
            nn.Conv2d(162, 243, (1, 3), (1, 2), (0, 1), 1),
            nn.BatchNorm2d(243),
            nn.LeakyReLU(),
            nn.Conv2d(243, 256, (1, 2), (1, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.moodLen = 16
        mood = np.random.normal(.0, 1., (moodSize, self.moodLen))
        if filterMood:
            mood = savgol_filter(mood, 129, 2, axis=0)
        self.mood = nn.Parameter(
            torch.from_numpy(mood).float().view(moodSize, self.moodLen).to(DEVICE),
            requires_grad=True
        )

        self.articulation = nn.Sequential(
            nn.Conv2d(256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1),
            nn.BatchNorm2d(256 + self.moodLen, .8),
            nn.LeakyReLU(),
            nn.Dropout2d(.2),
            nn.Conv2d(256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1),
            nn.BatchNorm2d(256 + self.moodLen, .8),
            nn.LeakyReLU(),
            nn.Dropout2d(.2),
            nn.Conv2d(256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1),
            nn.BatchNorm2d(256 + self.moodLen, .8),
            nn.LeakyReLU(),
            nn.Dropout2d(.2),
            nn.Conv2d(256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1),
            nn.BatchNorm2d(256 + self.moodLen, .8),
            nn.LeakyReLU(),
            nn.Dropout2d(.2),
            nn.Conv2d(256 + self.moodLen, 256 + self.moodLen, (4, 1), (4, 1), (1, 0), 1),
            nn.BatchNorm2d(256 + self.moodLen, .8),
            nn.LeakyReLU(),
            nn.Dropout2d(.2),
        )
        self.output = nn.Sequential(
            nn.Linear(256 + self.moodLen, 128),
            nn.Linear(128, self.shapesLen),
            nn.Tanh(),
        )

    def forward(self, inp, moodIndex=0, mood=None):
        out = self.formantAnalysis(inp)
        if mood is not None:
            out = torch.cat(
                (
                    out,
                    mood.view(
                        mood.view(-1, self.moodLen).size()[0], self.moodLen, 1, 1
                    ).expand(out.size()[0], self.moodLen, 64, 1)
                ),
                dim=1
            ).view(-1, 256 + self.moodLen, 64, 1)
        else:
            out = torch.cat(
                (
                    out,
                    torch.cat((
                        self.mood[moodIndex.chunk(chunks=1, dim=0)],
                        self.mood[(moodIndex + 1).chunk(chunks=1, dim=0)]
                    ), dim=0).view(
                        out.size()[0], self.moodLen, 1, 1
                    ).expand(out.size()[0], self.moodLen, 64, 1)
                ),
                dim=1
            ).view(-1, 256 + self.moodLen, 64, 1)
        out = self.articulation(out)
        out = self.output(out.view(-1, 256 + self.moodLen))
        return out.view(-1, self.shapesLen)


def train():
    batchSize = 10
    dataSet = Data()
    dataLoader = DataLoader(
        dataset=dataSet,
        batch_size=batchSize,
        shuffle=True
    )

    model = Model(dataSet.count).to(DEVICE)
    modelOptimizer = torch.optim.Adam(model.parameters())

    epochCount = 500

    runStr = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    logWriter = SummaryWriter(
        os.path.join(
            ROOT_PATH,
            'logs',
            runStr
        )
    )

    modelDir = os.path.join(
        ROOT_PATH,
        'model',
        runStr
    )
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    criterion = torch.nn.MSELoss().to(DEVICE)
    for epochIdx in range(epochCount):
        for i, inputData, target in dataLoader:
            # compensate for paired input
            inputData = inputData.view(-1, 1, 64, 32)
            target = target.view(-1, model.shapesLen)
            targetPairView = target.view(-1, 2, model.shapesLen)

            modelOptimizer.zero_grad()
            modelResult = model(inputData, i)
            modelPairView = modelResult.view(-1, 2, model.shapesLen)
            modelPairViewDetached = modelPairView.detach()

            modelResultPairView = modelResult.view(-1, 2, model.shapesLen)

            shapeLoss = criterion(
                modelResultPairView,
                targetPairView
            )

            motionLoss = criterion(
                modelResultPairView[:, 1, :] - modelResultPairView[:, 0, :],
                targetPairView[:, 1, :] - targetPairView[:, 0, :]
            )

            emotionLoss = criterion(
                model.mood[i],
                model.mood[i + 1]
            )

            (shapeLoss + motionLoss + emotionLoss).backward()
            modelOptimizer.step()

            logWriter.add_scalar('emotion', emotionLoss.item(), epochIdx + 1)
            logWriter.add_scalar('motion', motionLoss.item(), epochIdx + 1)
            logWriter.add_scalar('shape', shapeLoss.item(), epochIdx + 1)

        if (epochIdx + 1) % 10 == 0:
            figure, ax = plt.subplots()
            ax.set_ylim([-1, 1])
            for curves in targetPairView[:, 0, :].view(-1, model.shapesLen).cpu().numpy().T[23:]:
                ax.plot(curves)
            logWriter.add_figure('targetMouth', figure, epochIdx + 1)

            figure, ax = plt.subplots()
            ax.set_ylim([-1, 1])
            for curves in modelPairViewDetached[:, 0, :].view(-1, model.shapesLen).cpu().numpy().T[23:]:
                ax.plot(curves)
            logWriter.add_figure('resultMouth', figure, epochIdx + 1)

            figure, ax = plt.subplots()
            ax.set_ylim([-1, 1])
            for curves in targetPairView[:, 0, :].view(-1, model.shapesLen).cpu().numpy().T[:23]:
                ax.plot(curves)
            logWriter.add_figure('targetRest', figure, epochIdx + 1)

            figure, ax = plt.subplots()
            ax.set_ylim([-1, 1])
            for curves in modelPairViewDetached[:, 0, :].view(-1, model.shapesLen).cpu().numpy().T[:23]:
                ax.plot(curves)
            logWriter.add_figure('resultRest', figure, epochIdx + 1)

            figure, ax = plt.subplots()
            ax.set_ylim([-2, 2])
            plotLen = 72  # 3 seconds with 24 fps
            randomSection = random.randint(0, model.mood.size()[0] - plotLen + 1)
            for signal in model.mood.detach().cpu().numpy().T:
                ax.plot(signal[randomSection:randomSection + plotLen])
            logWriter.add_figure('mood', figure, epochIdx + 1)

        if (epochIdx + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(modelDir, '{}_E{:05d}.pth'.format(runStr, epochIdx + 1)))

    torch.save(model.state_dict(), os.path.join(modelDir, '{}_fin.pth'.format(runStr)))


if __name__ == '__main__':
    print('start: {}'.format(datetime.now()))
    start = datetime.now()
    print('training')
    train()
    print('done')
    print('duration: {}'.format(datetime.now() - start))
