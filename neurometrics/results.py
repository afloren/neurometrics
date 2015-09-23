import numpy
import scipy
import scipy.cluster
import scipy.signal
import mvpa2
import mvpa2.misc.io
import xml.etree.ElementTree

def tag_to_list(nodes,path,convert=float):
    return [convert(n.find(path).text) for n in nodes]

def filter_nodes(nodes,path,compare,convert=float):
    return [n for n in nodes if compare(convert(n.find(path).text))]

def filter_events_by_name(events,name):
    return filter_nodes(events,'name',lambda a: a == name,convert=str)

def filter_events_by_time(events,start,stop):
    return filter_nodes(events,'time',lambda a: start <= a <= stop)

def filter_events_by_block(events,start_event,duration):
    start = float(start_event.find('time').text)
    stop = start + duration
    return filter_events_by_time(events,start,stop)

def process_results_file(filename):
    tree = xml.etree.ElementTree.parse(filename)

    root = tree.getroot()

    results = root.findall('MrResults')

    num_frames = 30
    num_blocks = 6
    num_runs = len(results)
    total_blocks = num_runs * num_blocks
    total_frames = total_blocks * num_frames
    tr = 2.5

    block_foes = []
    block_mean_distance = []
    block_performance = []

    for r in results:
        searchTime = 45.0
        
        events = r.findall('events/*')
        search_events = filter_events_by_name(events,'SearchEvent')
        assert len(search_events) == num_blocks
        foe_events = filter_events_by_name(events,'FoeEvent')
        correct_events = filter_events_by_name(events,'CorrectEvent')
        
        foes = [filter_events_by_block(foe_events,s,searchTime) for s in search_events]
        block_foes += foes
        block_mean_distance += [numpy.mean(tag_to_list(f,'distance')) for f in foes]
        
        block_correct = [filter_events_by_block(correct_events,s,searchTime) for s in search_events]
        block_performance += [float(len(a))/float(len(b)) for a,b in zip(block_correct,foes)]

    p = numpy.polyfit(numpy.array(block_mean_distance),numpy.array(block_performance),1)
    block_attention = [(y - numpy.polyval(p,x)) for x,y in zip(block_mean_distance,block_performance)]

    cb,_ = scipy.cluster.vq.kmeans(numpy.array(block_mean_distance),3)
    cb.sort()
    quantized,_ = scipy.cluster.vq.vq(numpy.array(block_mean_distance),cb)
    block_quantized_distance = quantized.tolist()

    run = numpy.zeros(total_frames)
    cue = numpy.zeros(total_frames)
    move = numpy.zeros(total_frames)
    search = numpy.zeros(total_frames)
    control = numpy.zeros(total_frames)
    distance = numpy.zeros(total_frames)
    performance = numpy.zeros(total_frames)
    attention = numpy.zeros(total_frames)
    quantized_distance = numpy.zeros(total_frames)
    continuous_distance = numpy.zeros(total_frames)

    frame = 0
    
    for j,r in enumerate(results):
        searchTime = 45.0
        moveTime = float(r.find('config/moveTime/Value').text)
        cueEnabled = bool(r.find('config/cueTime/enabled').text)
        cueTime = float(r.find('config/cueTime/Value').text)
        controlEnabled = bool(r.find('config/distractors/enabled').text)
        controlTime = 15.0

        run[frame:frame+(num_frames*num_blocks)] = j
        for i in range(num_blocks):
            move[frame:frame+int(moveTime/tr)] = 1
            frame += int(moveTime/tr)
            if cueEnabled:
                cue[frame:frame+int(cueTime/tr)] = 1
                frame += int(cueTime/tr)
            search[frame:frame+int(searchTime/tr)] = 1
            distance[frame:frame+int(searchTime/tr)] = block_mean_distance[j * num_blocks + i]
            performance[frame:frame+int(searchTime/tr)] = block_performance[j * num_blocks + i]
            attention[frame:frame+int(searchTime/tr)] = block_attention[j * num_blocks + i]
            quantized_distance[frame:frame+int(searchTime/tr)] = block_quantized_distance[j * num_blocks + i]
            continuous_distance[frame:frame+int(searchTime/tr)] = scipy.signal.resample(tag_to_list(block_foes[j * num_blocks + i], 'distance'),
                                                                                        int(searchTime/tr),
                                                                                        tag_to_list(block_foes[j * num_blocks + i], 'time'))[0]
            frame += int(searchTime/tr)
            if controlEnabled:
                if cueEnabled:
                    cue[frame:frame+int(cueTime/tr)] = 1
                    frame += int(cueTime/tr)
                control[frame:frame+int(controlTime/tr)] = 1
                frame += int(controlTime/tr)
     
    attributes = dict()
    attributes['run'] = run
    attributes['cue'] = cue
    attributes['move'] = move
    attributes['search'] = search
    attributes['control'] = control
    attributes['distance'] = distance
    attributes['performance'] = performance
    attributes['attention'] = attention
    attributes['quantized_distance'] = quantized_distance
    attributes['continuous_distance'] = continuous_distance

    open('attributes.txt','w').close()
    cd = mvpa2.misc.io.ColumnData('attributes.txt')
    cd.update(attributes)
    cd.tofile('attributes.txt')


