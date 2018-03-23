
def makedata(s, n, l, filename)
	puts filename
	f = open(filename, "w")
	n.times do
		a = s * rand() / l
		b = rand() * l
		c = rand()
		data = []
		l.times do |i|
			data.push a * (i - b)**2 + c
		end
		ave = data.inject{|sum, v| sum + v} / data.size
		data.map{|v| v - ave}
		f.puts data.join(",")
	end
end

TRAIN_DATA = 1000
TEST_DATA = 100
L = 10

makedata(1, TRAIN_DATA, L, "on.txt")
makedata(-1, TRAIN_DATA, L, "off.txt")
makedata(1, TEST_DATA, L, "on_test.txt")
makedata(-1, TEST_DATA, L, "off_test.txt")
