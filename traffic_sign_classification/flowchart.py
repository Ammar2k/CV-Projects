import graphviz

dot = graphviz.Digraph(comment='Model Training Flowchart')

dot.node('A', 'Import and Preprocess Images')
dot.node('B', 'Split Train/Test/Validation Data')
dot.node('C', 'Define Model Architecture')
dot.node('D', 'Compile Model')
dot.node('E', 'Train Model')
dot.node('F', 'Evaluate Model')
dot.node('G', 'Plot Loss and Accuracy')
dot.node('H', 'Predict Images')

dot.edges([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'H')])

dot.render('model-training-flowchart.gv', view=True)
