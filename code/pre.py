import collections

total = collections.defaultdict(lambda: 0)
correct = collections.defaultdict(lambda: 0)
totalNum = 0
correctNum = 0
top3CorrectNum = 0
top5CorrectNum = 0
mrr = 0
with open('responsive_taggee_predict1', 'r') as inf, open('responsive_taggee_predict1_summary', 'w') as outf:
	for line in inf:
		line = line.split()
		if line[0] == 'Actual:':
			actual = line[1]
			predict = line[3]
			if actual == predict:
				correct[actual] += 1
				correctNum += 1
			total[actual] += 1
			totalNum += 1
			rank = int(line[5])
			if rank <= 3:
				top3CorrectNum += 1
			if rank <= 5:
				top5CorrectNum += 1
			mrr += 1.0/rank
	outf.write('test data size: ')
	outf.write(str(totalNum))
	outf.write('  number of taggees: ')
	outf.write(str(len(total)))
	outf.write('  accuracy: ')
	outf.write(str(correctNum/float(totalNum)))
	outf.write('  top 3 accuracy: ')
	outf.write(str(top3CorrectNum/float(totalNum)))
	outf.write('  top 5 accuracy: ')
	outf.write(str(top5CorrectNum/float(totalNum)))
	outf.write('  MRR: ')
	outf.write(str(mrr/totalNum))
	outf.write('\n')
	for taggee, value in sorted(total.items(), key=lambda k_v: k_v[1], reverse=True):
		outf.write(taggee)
		outf.write(': correct ')
		outf.write(str(correct[taggee]))
		outf.write(' out of ')
		outf.write(str(value))
		outf.write('\n')
		

