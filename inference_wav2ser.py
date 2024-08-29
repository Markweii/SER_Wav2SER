# from speechbrain.pretrained.interfaces import foreign_class
# classifier = foreign_class(source="Inference_wav2vec2", pymodule_file='hyperparams.yaml', savedir="Inference_wav2vec2")
# out_prob, score, index, text_lab = classifier.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")
# print(text_lab)

from speechbrain.pretrained.interfaces import foreign_class
import torchaudio
classifier = foreign_class(source="Inference", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

# Perform classification
audio_file = './data/Ang_ccc1.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
print(out_prob)
print(score)
print(index)
print(text_lab)
print('Emotion Predicted1: ' + text_lab[0])

# Perform classification
audio_file = './data/Ang_ccc2.wav'
# test_emotion: hap
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
print(out_prob)
print(score)
print(index)
print(text_lab)
print('Emotion Predicted2: ' + text_lab[0])

# Another speaker
audio_file = './data/Ang_ccc3.wav'
# test_emotion: sad
output_prob, score, index, text_lab = classifier.classify_file(audio_file)
print(output_prob)
print(score)
print(index)
print(text_lab)
print('Emotion Predicted3: ' + text_lab[0])

# Perform classification
audio_file = './data/Ang_ccc4.wav'
# test_emotion: angry
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
print(out_prob)
print('Emotion Predicted4: ' + text_lab[0])

audio_file = './data/Ang_ccc5.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted5: ' + text_lab[0])

audio_file = './data/Ang_t1.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted6: ' + text_lab[0])

audio_file = './data/Ang_t2.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted7: ' + text_lab[0])

audio_file = './data/Ang_t3.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted8: ' + text_lab[0])

audio_file = './data/Ang_t4.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted9: ' + text_lab[0])

audio_file = './data/Ang_z3.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted10: ' + text_lab[0])

audio_file = './data/Ang_z4.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
print(out_prob)
print('Emotion Predicted11: ' + text_lab[0])

audio_file = './data/Ang_z5.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted12: ' + text_lab[0])

audio_file = './data/Ang_z6.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted13: ' + text_lab[0])

audio_file = './data/Hap_z1.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted14: ' + text_lab[0])

audio_file = './data/Hap_z2.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted15: ' + text_lab[0])

audio_file = './data/Hap_z3.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted16: ' + text_lab[0])

audio_file = './data/Hap_z4.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
print('Emotion Predicted17: ' + text_lab[0])

# audio_file = './data/Neu_w4.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# # print(out_prob)
# print('Emotion Predicted18: ' + text_lab[0])

# audio_file = './data/Sad_c1.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# print(out_prob)
# print('Emotion Predicted19: ' + text_lab[0])

# audio_file = './data/Sad_c2.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# # print(out_prob)
# print('Emotion Predicted20: ' + text_lab[0])

# audio_file = './data/Sad_c3.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# # print(out_prob)
# print('Emotion Predicted21: ' + text_lab[0])

# audio_file = './data/Sad_c4.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# # print(out_prob)
# print('Emotion Predicted22: ' + text_lab[0])

# audio_file = './data/Sad_c5.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# # print(out_prob)
# print('Emotion Predicted23: ' + text_lab[0])

# audio_file = './data/Sad_w1.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# # print(out_prob)
# print('Emotion Predicted24: ' + text_lab[0])

# audio_file = './data/Sad_w2.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# # print(out_prob)
# print('Emotion Predicted25: ' + text_lab[0])

# audio_file = './data/Ang_cc1.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# # print(out_prob)
# print('Emotion Predicted26: ' + text_lab[0])

audio_file = './data/ang.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
print(output_prob)
print(score)
print(index)
print(text_lab)
print('Emotion Predicted27: ' + text_lab[0])

audio_file = './data/ang3.wav'
# test_emotion: neu
out_prob, score, index, text_lab = classifier.classify_file(audio_file)
print(output_prob)
print(score)
print(index)
print(text_lab)
print('Emotion Predicted28: ' + text_lab[0])

# audio_file = './data/ang5.wav'
# # test_emotion: neu
# out_prob, score, index, text_lab = classifier.classify_file(audio_file)
# # print(out_prob)
# print('Emotion Predicted29: ' + text_lab[0])