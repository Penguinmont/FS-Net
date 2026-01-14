"""
FS-Net TensorFlow 2.x 兼容版本
- 移除 tensorflow.contrib 依赖
- 使用 tf.keras 替代旧API
"""

import tensorflow as tf


class MultiBiGRU(tf.keras.layers.Layer):
    """多层双向GRU"""
    
    def __init__(self, hidden, layer, keep_prob, is_train, is_cat=True, **kwargs):
        super(MultiBiGRU, self).__init__(**kwargs)
        self._hidden = hidden
        self._layer = layer
        self._keep_prob = keep_prob
        self._is_train = is_train
        self._is_cat = is_cat
        
        # 创建多层双向GRU
        self.bi_grus = []
        for i in range(layer):
            forward_gru = tf.keras.layers.GRU(
                hidden, return_sequences=True, return_state=True,
                dropout=1-keep_prob if is_train else 0,
                name=f'gru_fw_{i}'
            )
            backward_gru = tf.keras.layers.GRU(
                hidden, return_sequences=True, return_state=True,
                go_backwards=True,
                dropout=1-keep_prob if is_train else 0,
                name=f'gru_bw_{i}'
            )
            self.bi_grus.append((forward_gru, backward_gru))
    
    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        outputs = [inputs]
        output_states = []
        
        for i, (gru_fw, gru_bw) in enumerate(self.bi_grus):
            # Forward
            fw_out, fw_state = gru_fw(outputs[-1], training=training)
            # Backward
            bw_out, bw_state = gru_bw(outputs[-1], training=training)
            # Reverse backward output
            bw_out = tf.reverse(bw_out, axis=[1])
            
            # Concatenate
            bi_out = tf.concat([fw_out, bw_out], axis=-1)
            bi_state = tf.concat([fw_state, bw_state], axis=-1)
            
            outputs.append(bi_out)
            output_states.append(bi_state)
        
        if self._is_cat:
            res = tf.concat(outputs[1:], axis=-1)
            res_state = tf.concat(output_states, axis=-1)
        else:
            res = outputs[-1]
            res_state = output_states[-1]
        
        return res_state, res


class FSNet(tf.keras.Model):
    """FS-Net 流量分类模型 - TensorFlow 2.x版本"""
    
    def __init__(self, config, **kwargs):
        super(FSNet, self).__init__(**kwargs)
        self.config = config
        
        # 类别权重
        if hasattr(config, 'class_weights') and config.class_weights is not None:
            self.class_weights = tf.constant(config.class_weights, dtype=tf.float32)
        else:
            self.class_weights = None
        
        # Embedding层
        self.embedding = tf.keras.layers.Embedding(
            config.length_num, config.length_dim, name='length_embedding'
        )
        
        # Encoder
        self.encoder_dropout = tf.keras.layers.Dropout(1 - config.keep_prob)
        self.encoder_gru = MultiBiGRU(
            config.hidden, config.layer, config.keep_prob, 
            is_train=True, is_cat=True, name='encoder'
        )
        
        # Decoder
        self.decoder_gru = MultiBiGRU(
            config.hidden, config.layer, config.keep_prob,
            is_train=True, is_cat=True, name='decoder'
        )
        
        # Reconstruction
        self.rec_dense1 = tf.keras.layers.Dense(config.hidden, activation='selu', name='rec_dense1')
        self.rec_dense2 = tf.keras.layers.Dense(config.length_num, name='rec_dense2')
        
        # Compression
        self.compress_dense = tf.keras.layers.Dense(
            2 * config.hidden, activation='selu', name='compress'
        )
        self.compress_dropout = tf.keras.layers.Dropout(1 - config.keep_prob)
        
        # Classifier
        self.classifier = tf.keras.layers.Dense(config.class_num, name='classifier')
    
    def call(self, inputs, training=False):
        flow, mask = inputs
        
        # Embedding
        seq = self.embedding(flow)
        
        # Encoder
        if training:
            seq = self.encoder_dropout(seq, training=training)
        e_fea, enc_outputs = self.encoder_gru(seq, training=training)
        
        # Decoder input: broadcast encoder feature
        max_len = tf.shape(flow)[1]
        dec_input = tf.tile(tf.expand_dims(e_fea, axis=1), [1, max_len, 1])
        
        # Decoder
        d_fea, dec_outputs = self.decoder_gru(dec_input, training=training)
        
        # Reconstruction
        rec_logits = self.rec_dense1(dec_outputs)
        rec_logits = self.rec_dense2(rec_logits)
        
        # Feature fusion
        feature = tf.concat([e_fea, d_fea], axis=-1)
        feature = self.compress_dense(feature)
        if training:
            feature = self.compress_dropout(feature, training=training)
        
        # Classification
        logits = self.classifier(feature)
        pred = tf.argmax(logits, axis=-1)
        
        return logits, pred, rec_logits
    
    def compute_loss(self, flow, labels, mask, logits, rec_logits):
        """计算总损失"""
        # 分类损失
        if self.class_weights is not None:
            sample_weights = tf.gather(self.class_weights, labels)
            c_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
            c_loss = tf.reduce_sum(c_loss * sample_weights) / tf.reduce_sum(sample_weights)
        else:
            c_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            )
        
        # 重构损失
        rec_logits_flat = tf.reshape(rec_logits, [-1, self.config.length_num])
        flow_flat = tf.reshape(flow, [-1])
        mask_flat = tf.cast(tf.reshape(mask, [-1]), tf.float32)
        
        rec_loss_all = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=flow_flat, logits=rec_logits_flat
        )
        rec_loss = tf.reduce_sum(rec_loss_all * mask_flat) / (tf.reduce_sum(mask_flat) + 1e-8)
        
        # 总损失
        total_loss = c_loss + self.config.rec_loss * rec_loss
        
        return total_loss, c_loss, rec_loss


def create_model(config):
    """创建模型"""
    return FSNet(config)
