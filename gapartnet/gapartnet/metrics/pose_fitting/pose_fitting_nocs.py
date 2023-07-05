'''
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
RANSAC for Similarity Transformation Estimation

Written by Srinath Sridhar
'''

import numpy as np
import cv2
import itertools

def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False):
    if source.shape[0]==1:
        source = [source[0],source[0]]
        target = [target[0],target[0]]
        
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    # print(SourceHom.shape)
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))
    # print(SourceHom)
    # Auto-parameter selection based on source-target heuristics
    TargetNorm = np.mean(np.linalg.norm(target, axis=1))
    SourceNorm = np.mean(np.linalg.norm(source, axis=1))
    RatioTS = (TargetNorm / SourceNorm)
    RatioST = (SourceNorm / TargetNorm)
    PassT = RatioST if(RatioST>RatioTS) else RatioTS
    # print(TargetNorm,SourceNorm,RatioTS,RatioST,PassT)
    StopT = 0.5 #PassT / 100
    nIter = 100
    if verbose:
        print('Pass threshold: ', PassT)
        print('Stop threshold: ', StopT)
        print('Number of iterations: ', nIter)

    SourceInliersHom, TargetInliersHom, BestInlierRatio, BestInlierIdx = \
        getRANSACInliers(SourceHom, TargetHom, MaxIterations=nIter, PassThreshold=PassT, StopThreshold=StopT)
    # print("###################")
    # print(len(BestInlierIdx))
    
    # print("###################")
    # print(SourceInliersHom)
    if(BestInlierRatio < 0.01): # haoran: 0.1->0.01
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, np.array([None,None,None,]), None, None, None

    Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scales:', Scales)

    return Scales, Rotation, Translation, OutTransform, BestInlierIdx

def estimateRestrictedAffineTransform(source: np.array, target: np.array, verbose=False):
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    RetVal, AffineTrans, Inliers = cv2.estimateAffine3D(source, target)
    # We assume no shear in the affine matrix and decompose into rotation, non-uniform scales, and translation
    Translation = AffineTrans[:3, 3]
    NUScaleRotMat = AffineTrans[:3, :3]
    # NUScaleRotMat should be the matrix SR, where S is a diagonal scale matrix and R is the rotation matrix (equivalently RS)
    # Let us do the SVD of NUScaleRotMat to obtain R1*S*R2 and then R = R1 * R2
    R1, ScalesSorted, R2 = np.linalg.svd(NUScaleRotMat, full_matrices=True)

    if verbose:
        print('-----------------------------------------------------------------------')
    # Now, the scales are sort in ascending order which is painful because we don't know the x, y, z scales
    # Let's figure that out by evaluating all 6 possible permutations of the scales
    ScalePermutations = list(itertools.permutations(ScalesSorted))
    MinResidual = 1e8
    Scales = ScalePermutations[0]
    OutTransform = np.identity(4)
    Rotation = np.identity(3)
    for ScaleCand in ScalePermutations:
        CurrScale = np.asarray(ScaleCand)
        CurrTransform = np.identity(4)
        CurrRotation = (np.diag(1 / CurrScale) @ NUScaleRotMat).transpose()
        CurrTransform[:3, :3] = np.diag(CurrScale) @ CurrRotation
        CurrTransform[:3, 3] = Translation
        # Residual = evaluateModel(CurrTransform, SourceHom, TargetHom)
        Residual = evaluateModelNonHom(source, target, CurrScale,CurrRotation, Translation)
        if verbose:
            # print('CurrTransform:\n', CurrTransform)
            print('CurrScale:', CurrScale)
            print('Residual:', Residual)
            print('AltRes:', evaluateModelNoThresh(CurrTransform, SourceHom, TargetHom))
        if Residual < MinResidual:
            MinResidual = Residual
            Scales = CurrScale
            Rotation = CurrRotation
            OutTransform = CurrTransform

    if verbose:
        print('Best Scale:', Scales)

    if verbose:
        print('Affine Scales:', Scales)
        print('Affine Translation:', Translation)
        print('Affine Rotation:\n', Rotation)
        print('-----------------------------------------------------------------------')

    return Scales, Rotation, Translation, OutTransform

def getRANSACInliers(SourceHom, TargetHom, MaxIterations=100, PassThreshold=200, StopThreshold=0.5):
    BestResidual = 1e10
    BestInlierRatio = 0
    BestInlierIdx = np.arange(SourceHom.shape[1])
    for i in range(0, MaxIterations):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(SourceHom.shape[1], size=5)
        _, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        Residual, InlierRatio, InlierIdx = evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold)
        if Residual < BestResidual:
            BestResidual = Residual
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if BestResidual < StopThreshold:
            break

        # print('Iteration: ', i)
        # print('Residual: ', Residual)
        # print('InlierIdx: ', InlierIdx)

    return SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio, BestInlierIdx

def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    InlierIdx = np.where(ResidualVec < PassThreshold)
    nInliers = np.count_nonzero(InlierIdx)
    InlierRatio = nInliers / SourceHom.shape[1]
    return Residual, InlierRatio, InlierIdx[0]

def evaluateModelNoThresh(OutTransform, SourceHom, TargetHom):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def evaluateModelNonHom(source, target, Scales, Rotation, Translation):
    RepTrans = np.tile(Translation, (source.shape[0], 1))
    TransSource = (np.diag(Scales) @ Rotation @ source.transpose() + RepTrans.transpose()).transpose()
    Diff = target - TransSource
    ResidualVec = np.linalg.norm(Diff, axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def testNonUniformScale(SourceHom, TargetHom):
    OutTransform = np.matmul(TargetHom, np.linalg.pinv(SourceHom))
    ScaledRotation = OutTransform[:3, :3]
    Translation = OutTransform[:3, 3]
    Sx = np.linalg.norm(ScaledRotation[0, :])
    Sy = np.linalg.norm(ScaledRotation[1, :])
    Sz = np.linalg.norm(ScaledRotation[2, :])
    Rotation = np.vstack([ScaledRotation[0, :] / Sx, ScaledRotation[1, :] / Sy, ScaledRotation[2, :] / Sz])
    print('Rotation matrix norm:', np.linalg.norm(Rotation))
    Scales = np.array([Sx, Sy, Sz])

    # # Check
    # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    # Residual = np.linalg.norm(Diff[:3, :], axis=0)
    return Scales, Rotation, Translation, OutTransform

def estimateSimilarityUmeyama(SourceHom, TargetHom):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints
    # print(CenteredTarget, CenteredSource)
    # print(CovMatrix)
    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    Rotation = np.matmul(U, Vh).T # Transpose is the one that works

    varP = np.var(SourceHom[:3, :], axis=1).sum()
    ScaleFact = 1/varP * np.sum(D) # scale factor
    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)

    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)

    OutTransform = np.identity(4)
    OutTransform[:3, :3] = ScaleMatrix @ Rotation
    OutTransform[:3, 3] = Translation

    # # Check
    # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    # Residual = np.linalg.norm(Diff[:3, :], axis=0)
    return Scales, Rotation, Translation, OutTransform

def estimate3dbbox(coord, npcs_pred, part_mask):
    coord_part = coord[np.where(part_mask == True)]
    npcs_part = npcs_pred[np.where(part_mask == True)]

    coord_part=np.array(coord_part)
    npcs_part=np.array(npcs_part)

    Scales, Rotation, Translation, OutTransform, BestInlierIdx = estimateSimilarityTransform(npcs_part,coord_part)
    scale_pred = np.max(abs(npcs_part[BestInlierIdx]), axis=0)

    bbox_raw = np.array([
                [-scale_pred[0], -scale_pred[1], -scale_pred[2]],
                [scale_pred[0], -scale_pred[1], -scale_pred[2]],
                [-scale_pred[0], scale_pred[1], -scale_pred[2]],
                [-scale_pred[0], -scale_pred[1], scale_pred[2]],
                [scale_pred[0], scale_pred[1], -scale_pred[2]],
                
                [scale_pred[0], -scale_pred[1], scale_pred[2]],
                [-scale_pred[0], scale_pred[1], scale_pred[2]],
                [scale_pred[0], scale_pred[1], scale_pred[2]]
        ])
    bbox_trans = np.dot((bbox_raw * Scales[0]) , Rotation)  + Translation

    return bbox_trans, Scales, Rotation, Translation

def estimate3dbbox_part(coord_part, npcs_pred):
    Scales, Rotation, Translation, OutTransform, BestInlierIdx = estimateSimilarityTransform(npcs_pred,coord_part)
    if Rotation.all()==None:
        return np.array([None,None,None,]),None,None,None
    scale_pred = np.max(abs(npcs_pred[BestInlierIdx]), axis=0)

    bbox_raw = np.array([
                [-scale_pred[0], -scale_pred[1], -scale_pred[2]],
                [scale_pred[0], -scale_pred[1], -scale_pred[2]],
                [-scale_pred[0], scale_pred[1], -scale_pred[2]],
                [-scale_pred[0], -scale_pred[1], scale_pred[2]],
                [scale_pred[0], scale_pred[1], -scale_pred[2]],
                
                [scale_pred[0], -scale_pred[1], scale_pred[2]],
                [-scale_pred[0], scale_pred[1], scale_pred[2]],
                [scale_pred[0], scale_pred[1], scale_pred[2]]
        ])
    bbox_trans = np.dot((bbox_raw * Scales[0]) , Rotation)  + Translation

    return bbox_trans, Scales, Rotation, Translation