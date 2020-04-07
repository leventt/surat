import os
import random
from datetime import datetime
import numpy as np
from scipy.signal import savgol_filter
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from importlib.machinery import SourceFileLoader


ROOT_PATH = os.getenv('SURAT_ROOT_PATH', False)
if not ROOT_PATH:
    ROOT_PATH = os.path.dirname(__file__)
DEVICE = torch.device('cuda')
OUTPUT_COUNT = 8320 * 3  # 8320 vertex positions in 3 dimentions
INPUT_VALUES_PRECALC_PATH = os.path.join(ROOT_PATH, 'inputValues.precalc')

lpc = SourceFileLoader(
    'lpc',
    os.path.join(
        ROOT_PATH, 'LPCTorch/lpctorch/lpc.py'
    )
).load_module()


class Data(Dataset):
    def __init__(self, transforms=None, shiftRandom=True, validationAudioPath=None):
        self.transforms = transforms
        self.preview = validationAudioPath is not None
        self.shiftRandom = shiftRandom and not self.preview
        self.count = None

        animFPS = 29.97  # samSoar recorded with an ipad

        if self.preview:
            inputSpeechPath = validationAudioPath
        else:
            inputSpeechPath = os.path.join(ROOT_PATH, 'data', 'samSoar', 'samSoar.wav')
        self.waveform, self.sampleRate = torchaudio.load(inputSpeechPath)
        if self.sampleRate != 16000:
            self.waveform = torchaudio.transforms.Resample(self.sampleRate, 16000)(self.waveform)
            self.sampleRate = 16000

        self.count = int(animFPS * (self.waveform.size()[1] / self.sampleRate))

        self.LPC = lpc.LPCCoefficients(
            self.sampleRate,
            .032,
            .5,
            order=31  # 32 - 1
        )

        if os.path.exists(INPUT_VALUES_PRECALC_PATH):
            self.inputValues = torch.load(INPUT_VALUES_PRECALC_PATH)
        else:
            print('pre-calculating input values...')
            self.inputValues = torch.Tensor([])
            audioFrameLen = int(.016 * 16000 * (64 + 1))
            audioHalfFrameLen = int(audioFrameLen / 2.)
            for i in range(self.count):
                print('{}/{}'.format(i + 1, self.count))
                audioRoll = -1 * (int(self.waveform.size()[1] / self.count) - audioHalfFrameLen)
                audioIdxRoll = int(i * audioRoll)
                audioIdxRollPair = int((i + 1) * audioRoll)

                self.inputValues = torch.cat(
                    (
                        self.inputValues,
                        torch.cat(
                            (
                                self.LPC(
                                    torch.roll(self.waveform[0:1, :], audioIdxRoll, dims=0)[:, :audioFrameLen]
                                ).view(1, 1, 64, 32),
                                self.LPC(
                                    torch.roll(self.waveform[0:1, :], audioIdxRollPair, dims=0)[:, :audioFrameLen]
                                ).view(1, 1, 64, 32)
                            ),
                            dim=0,
                        ).view(2, 1, 64, 32)
                    ), dim=0
                ).view(-1, 1, 64, 32)
            self.inputValues = self.inputValues.view(-1, 2, 1, 64, 32)
            torch.save(self.inputValues, INPUT_VALUES_PRECALC_PATH)

    def __getitem__(self, i):
        if i < 0:  # for negative indexing
            i = self.count + i

        inputValue = self.inputValues[i]

        if self.preview:
            return (
                torch.Tensor([i]).long(),
                inputValue[0],
                torch.zeros((1, OUTPUT_COUNT))
            )

        targetValue = torch.from_numpy(np.append(
            np.load(
                os.path.join(
                    ROOT_PATH,
                    'data', 'samSoar', 'maskSeq',
                    'mask.{:05d}.npy'.format(i + 1)
                )
            ),
            np.load(
                os.path.join(
                    ROOT_PATH,
                    'data', 'samSoar', 'maskSeq',
                    'mask.{:05d}.npy'.format(i + 2)
                )
            )
        )).float().view(-1, OUTPUT_COUNT)

        return (
            torch.Tensor([i]).long(),
            inputValue,
            # output values are assumed to have max of 2 and min of -2
            (targetValue) * .5
        )

    def __len__(self):
        if self.preview:
            return self.count
        return self.count - 1  # for pairs

class Model(nn.Module):
    def __init__(self, moodSize, filterMood=False):
        super(Model, self).__init__()

        self.formantAnalysis = nn.Sequential(
            nn.Conv2d(1, 72, (1, 3), (1, 2), (0, 1), 1),
            nn.LeakyReLU(),
            nn.Conv2d(72, 108, (1, 3), (1, 2), (0, 1), 1),
            nn.LeakyReLU(),
            nn.Conv2d(108, 162, (1, 3), (1, 2), (0, 1), 1),
            nn.LeakyReLU(),
            nn.Conv2d(162, 243, (1, 3), (1, 2), (0, 1), 1),
            nn.LeakyReLU(),
            nn.Conv2d(243, 256, (1, 2), (1, 2)),
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
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (3, 1), (2, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                256 + self.moodLen, 256 + self.moodLen, (4, 1), (4, 1), (1, 0), 1
            ),
            nn.LeakyReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(256 + self.moodLen, 150),
            nn.Linear(150, OUTPUT_COUNT),
            nn.Tanh(),
        )

    def forward(self, inp, mood, moodIndex=0):
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
        return out.view(-1, OUTPUT_COUNT)


def train():
    batchSize = 1024
    dataSet = Data()
    dataLoader = DataLoader(
        dataset=dataSet,
        batch_size=batchSize,
        shuffle=True,
        num_workers=8
    )

    model = Model(dataSet.count, filterMood=False).to(DEVICE)
    modelOptimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2
    )

    epochCount = 200000

    runStr = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    logWriter = SummaryWriter(os.path.join(ROOT_PATH, 'logs', runStr))

    modelDir = os.path.join(ROOT_PATH, 'model', runStr)
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    criterion = torch.nn.MSELoss().to(DEVICE)
    for epochIdx in range(epochCount):
        for i, inputData, target in dataLoader:
            i = i.to(DEVICE)
            inputData = inputData.to(DEVICE)
            target = target.to(DEVICE)
            # compensate for paired input
            inputData = inputData.view(-1, 1, 64, 32)
            target = target.view(-1, OUTPUT_COUNT)
            targetPairView = target.view(-1, 2, OUTPUT_COUNT)

            modelOptimizer.zero_grad()
            modelResult = model(inputData, None, i)
            modelResultPairView = modelResult.view(-1, 2, OUTPUT_COUNT)

            shapeLoss = criterion(
                modelResultPairView,
                targetPairView
            )

            motionLoss = criterion(
                100 * (modelResultPairView[:, 1, :] - modelResultPairView[:, 0, :]),
                100 * (targetPairView[:, 1, :] - targetPairView[:, 0, :]),
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

        if (epochIdx + 1) % 50 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(modelDir, '{}_E{:05d}.pth'.format(runStr, epochIdx + 1)),
            )

    torch.save(model.state_dict(), os.path.join(modelDir, '{}_fin.pth'.format(runStr)))



if __name__ == '__main__':
    print('start: {}'.format(datetime.now()))
    start = datetime.now()
    print('training')
    train()
    print('done')
    print('duration: {}'.format(datetime.now() - start))
