from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty
import torch.nn.functional as F
import tqdm

def run_eval_per_frame(
    model,
    batch,
    get_trackwise_metrics=True,
    radius=8,
    threshold=0.5,
    use_certainty=False,
):
  with torch.no_grad():
    pred_tracks, track_logits, visible_logits, tracking_state = model(
        video=batch['video'][:, :1], query_points=batch['query_points']
    )
    pred_visible = visible_logits > 0
    pred_tracks, pred_visible = [pred_tracks.cpu()], [pred_visible.cpu()]
    pred_track_logits, pred_visible_logits = [track_logits.cpu()], [
        visible_logits.cpu()
    ]
    for frame in tqdm.tqdm(range(1, batch['video'].shape[1])):
      # ***************************************************
      # HERE WE RUN POINT TRACKING IN PURELY ONLINE FASHION
      # ***************************************************
      (
          curr_tracks,
          curr_track_logits,
          curr_visible_logits,
          tracking_state,
      ) = model(
          video=batch['video'][:, frame : frame + 1],
          state=tracking_state,
      )
      curr_visible = curr_visible_logits > 0
      # ***************************************************
      pred_tracks.append(curr_tracks.cpu())
      pred_visible.append(curr_visible.cpu())
      pred_track_logits.append(curr_track_logits.cpu())
      pred_visible_logits.append(curr_visible_logits.cpu())
    tracks = torch.cat(pred_tracks, dim=1).transpose(1, 2)
    pred_visible = torch.cat(pred_visible, dim=1).transpose(1, 2)
    track_logits = torch.cat(pred_track_logits, dim=1).transpose(1, 2)
    visible_logits = torch.cat(pred_visible_logits, dim=1).transpose(1, 2)

    pred_certainty = tracker_certainty(tracks, track_logits, radius)
    pred_visible_and_certain = (
        F.sigmoid(visible_logits) * pred_certainty
    ) > threshold

    if use_certainty:
      occluded = ~(pred_visible_and_certain.squeeze(-1))
    else:
      occluded = ~(pred_visible.squeeze(-1))

  scalars = evaluation_datasets.compute_tapvid_metrics(
      batch['query_points'].cpu().numpy(),
      batch['occluded'].cpu().numpy(),
      batch['target_points'].cpu().numpy(),
      occluded.numpy() + 0.0,
      tracks.numpy()[..., ::-1],
      query_mode='first',
      get_trackwise_metrics=get_trackwise_metrics,
  )
  return (
      tracks.numpy()[..., ::-1],
      occluded.numpy(),
      {k: v.sum(0) for k, v in scalars.items()},
  )


# @title Function for raw data to the input format {form-width: "25%"}
def deterministic_eval(cached_dataset, strided=False):
  if not strided:
    for sample in cached_dataset:
      batch = sample['davis'].copy()
      # batch['video'] = (batch['video'] + 1) / 2
      batch['visible'] = np.logical_not(batch['occluded'])[..., None]
      batch['padding'] = np.ones(
          batch['query_points'].shape[:2], dtype=np.bool_
      )
      batch['loss_mask'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )
      batch['appearance'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )

      yield batch
  else:
    for sample in cached_dataset:
      batch = sample['davis'].copy()
      # batch['video'] = (batch['video'] + 1) / 2
      batch['visible'] = np.logical_not(batch['occluded'])[..., None]
      batch['padding'] = np.ones(
          batch['query_points'].shape[:2], dtype=np.bool_
      )
      batch['loss_mask'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )
      batch['appearance'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )
      backward_batch = {k: v.copy() for k, v in batch.items()}
      for key in ['visible', 'appearance', 'loss_mask', 'target_points']:
        backward_batch[key] = np.flip(backward_batch[key], axis=2)
      backward_batch['video'] = np.flip(backward_batch['video'], axis=1)
      backward_queries = (
          backward_batch['video'].shape[1]
          - backward_batch['query_points'][..., 0]
          - 1
      )
      backward_batch['query_points'][..., 0] = backward_queries
      yield batch, backward_batch


def get_model():
    model = TAPNext(image_size=(256, 256))
    ckpt_path = 'bootstapnext_ckpt.npz'
    model = restore_model_from_jax_checkpoint(model, ckpt_path)
    model.cuda()
    return model