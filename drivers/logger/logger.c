/*
 * partlog.c — Persistent partition log backend for Lumia 950 XL (kernel 3.10)
 *
 * Opens the target partition purely through the kernel block subsystem —
 * no /dev paths, no udev, no devtmpfs required.
 *
 * Discovery strategy
 * ──────────────────────────────────────────────────────────────────────────
 * We register a struct class_interface on the "block" class.
 * The block layer calls .add_dev() for every block device (disk + partition)
 * as it comes online — including any that were already registered before us.
 * We match on the GPT partition label (CONFIG_PARTLOG_PART_LABEL) or fall
 * back to disk name + partition number.
 *
 * Logging levels used
 * ──────────────────────────────────────────────────────────────────────────
 * pr_err   — hard failures (I/O errors, alloc failures, bad header)
 * pr_warn  — soft failures (ring full → dropped bytes)
 * pr_info  — lifecycle events (attach, detach, format, wrap, panic flush)
 * pr_debug — per-I/O tracing (bio submit, flush stats); silent unless
 *            CONFIG_DYNAMIC_DEBUG or pr_debug is enabled for this module
 *
 * To enable debug output at runtime (CONFIG_DYNAMIC_DEBUG=y):
 *   echo "module partlog +p" > /sys/kernel/debug/dynamic_debug/control
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/console.h>
#include <linux/spinlock.h>
#include <linux/mutex.h>
#include <linux/workqueue.h>
#include <linux/slab.h>
#include <linux/bio.h>
#include <linux/blkdev.h>
#include <linux/genhd.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/notifier.h>
#include <linux/atomic.h>
#include <linux/completion.h>
#include <linux/string.h>
#include <asm/page.h>

#define PARTLOG_TAG "partlog: "

/* ── tunables ─────────────────────────────────────────────────────────── */

static char partlog_label[64] = CONFIG_PARTLOG_PART_LABEL;
module_param_string(label, partlog_label, sizeof(partlog_label), 0444);
MODULE_PARM_DESC(label, "GPT partition label (preferred identification method)");

static char partlog_disk[32] = CONFIG_PARTLOG_DISK_NAME;
module_param_string(disk, partlog_disk, sizeof(partlog_disk), 0444);
MODULE_PARM_DESC(disk, "Fallback disk base name, e.g. mmcblk0");

static unsigned int partlog_part_nr = CONFIG_PARTLOG_PART_NR;
module_param_named(part_nr, partlog_part_nr, uint, 0444);
MODULE_PARM_DESC(part_nr, "Fallback partition number on that disk");

static unsigned int partlog_size_mb = CONFIG_PARTLOG_SIZE_MB;
module_param_named(size_mb, partlog_size_mb, uint, 0444);
MODULE_PARM_DESC(size_mb, "Partition size in MiB (must match real partition)");

/* ── on-disk structures ───────────────────────────────────────────────── */

#define PARTLOG_MAGIC      0x504C4F47UL
#define PARTLOG_VERSION    1
#define PARTLOG_DATA_START 512          /* header occupies first 512 bytes */

struct partlog_header {
    __le32  magic;
    __le32  version;
    __le64  boot_count;
    __le64  write_head;   /* byte offset within the data area */
    __le64  data_size;    /* total bytes in the data area     */
    __u8    _pad[512 - 4 - 4 - 8 - 8 - 8];
} __packed;

#define PARTLOG_REC_MAGIC  0xC0DE
struct partlog_rec_hdr {
    __le16  magic;
    __le16  len;
} __packed;
#define PARTLOG_REC_HDR_SZ sizeof(struct partlog_rec_hdr)

/* ── staging ring ────────────────────────────────────────────────────── */

#define RING_SIZE  (256 * 1024)
#define RING_MASK  (RING_SIZE - 1)

struct partlog_ring {
    spinlock_t   lock;
    char         buf[RING_SIZE];
    unsigned int head;
    unsigned int tail;
    unsigned int used;
    unsigned long long total_dropped;   /* lifetime bytes dropped */
};

/* ── driver state ────────────────────────────────────────────────────── */

struct partlog_state {
    struct block_device    *bdev;
    struct partlog_header   hdr;
    struct partlog_ring     ring;
    struct work_struct      flush_work;
    struct workqueue_struct *wq;
    void                   *io_buf;        /* one PAGE_SIZE bounce buffer */
    atomic_t                initialised;
    atomic_t                flush_pending;
    unsigned long long      total_written; /* lifetime bytes written to disk */
    unsigned int            wrap_count;    /* how many times partition wrapped */
};

static struct partlog_state *g_state;
static DEFINE_MUTEX(partlog_open_mutex);

/* ── submit_bio_wait compat ───────────────────────────────────────────── */

struct pl_bio_wait {
    struct completion done;
    int               error;
};

static void pl_bio_end_io(struct bio *bio, int error)
{
    struct pl_bio_wait *w = bio->bi_private;
    w->error = error;
    complete(&w->done);
}

static int pl_submit_bio_wait(int rw, struct bio *bio)
{
    struct pl_bio_wait w;
    init_completion(&w.done);
    bio->bi_private = &w;
    bio->bi_end_io  = pl_bio_end_io;
    submit_bio(rw, bio);
    wait_for_completion(&w.done);
    return w.error;
}

/* ── block I/O helpers ───────────────────────────────────────────────── */

static int partlog_sync_rw(struct partlog_state *s, loff_t off,
                            void *buf, size_t len, int write)
{
    size_t done = 0;
    int ret = 0;

    pr_debug(PARTLOG_TAG "%s off=%lld len=%zu\n",
             write ? "write" : "read", (long long)off, len);

    while (done < len) {
        loff_t    cur    = off + done;
        sector_t  sec    = cur >> 9;
        unsigned  pg_off = cur & (PAGE_SIZE - 1);
        size_t    pg_len = min_t(size_t, PAGE_SIZE - pg_off, len - done);
        struct bio *bio;

        if (write)
            memcpy((char *)s->io_buf + pg_off,
                   (const char *)buf + done, pg_len);

        bio = bio_alloc(GFP_NOIO, 1);
        if (!bio) {
            pr_err(PARTLOG_TAG "bio_alloc failed (off=%lld, done=%zu)\n",
                   (long long)off, done);
            ret = -ENOMEM;
            break;
        }

        bio->bi_bdev   = s->bdev;
        bio->bi_sector = sec;
        bio_add_page(bio, virt_to_page(s->io_buf), pg_len, pg_off);

        pr_debug(PARTLOG_TAG "bio %s sector=%lu pg_off=%u pg_len=%zu\n",
                 write ? "WR" : "RD", (unsigned long)sec, pg_off, pg_len);

        ret = pl_submit_bio_wait(write ? WRITE_SYNC : READ_SYNC, bio);
        bio_put(bio);

        if (ret) {
            pr_err(PARTLOG_TAG "bio %s error %d (sector=%lu)\n",
                   write ? "write" : "read", ret, (unsigned long)sec);
            break;
        }

        if (!write)
            memcpy((char *)buf + done,
                   (char *)s->io_buf + pg_off, pg_len);

        done += pg_len;
    }

    if (!ret && done)
        pr_debug(PARTLOG_TAG "%s complete: %zu bytes\n",
                 write ? "write" : "read", done);

    return ret;
}

#define partlog_read(s, off, buf, len)  partlog_sync_rw(s, off, buf, len, 0)
#define partlog_write(s, off, buf, len) partlog_sync_rw(s, off, buf, len, 1)

static int partlog_write_header(struct partlog_state *s)
{
    int ret = partlog_write(s, 0, &s->hdr, sizeof(s->hdr));
    if (ret)
        pr_err(PARTLOG_TAG "header write failed: %d\n", ret);
    else
        pr_debug(PARTLOG_TAG "header updated: boot=%llu head=%llu\n",
                 (unsigned long long)le64_to_cpu(s->hdr.boot_count),
                 (unsigned long long)le64_to_cpu(s->hdr.write_head));
    return ret;
}

/* ── ring helpers ────────────────────────────────────────────────────── */

static void ring_push(struct partlog_ring *r, const char *buf, size_t len)
{
    size_t avail = RING_SIZE - r->used;
    size_t n     = min(len, avail);
    size_t p1    = min(n, (size_t)(RING_SIZE - r->head));

    if (n < len) {
        size_t dropped = len - n;
        r->total_dropped += dropped;
        pr_warn_ratelimited(PARTLOG_TAG
            "ring full — dropped %zu bytes (used=%u/%u, lifetime=%llu)\n",
            dropped, r->used, RING_SIZE, r->total_dropped);
    }

    if (!n)
        return;

    memcpy(r->buf + r->head, buf, p1);
    if (n - p1)
        memcpy(r->buf, buf + p1, n - p1);
    r->head  = (r->head + n) & RING_MASK;
    r->used += n;
}

static size_t ring_pop(struct partlog_ring *r, char *out, size_t max)
{
    size_t n  = min(r->used, max);
    size_t p1 = min(n, (size_t)(RING_SIZE - r->tail));
    memcpy(out, r->buf + r->tail, p1);
    if (n - p1)
        memcpy(out + p1, r->buf, n - p1);
    r->tail   = (r->tail + n) & RING_MASK;
    r->used  -= n;
    return n;
}

/* ── console .write ──────────────────────────────────────────────────── */

static void partlog_console_write(struct console *con,
                                   const char *text, unsigned int len)
{
    struct partlog_state *s = g_state;
    struct partlog_rec_hdr rec;
    unsigned long flags;

    if (!s || !atomic_read(&s->initialised))
        return;

    rec.magic = cpu_to_le16(PARTLOG_REC_MAGIC);
    rec.len   = cpu_to_le16((u16)min_t(unsigned int, len, 0xFFFFu));

    spin_lock_irqsave(&s->ring.lock, flags);
    ring_push(&s->ring, (const char *)&rec, PARTLOG_REC_HDR_SZ);
    ring_push(&s->ring, text, le16_to_cpu(rec.len));
    spin_unlock_irqrestore(&s->ring.lock, flags);

    if (!atomic_xchg(&s->flush_pending, 1))
        queue_work(s->wq, &s->flush_work);
}

static struct console partlog_console = {
    .name  = "partlog",
    .write = partlog_console_write,
    .flags = CON_PRINTBUFFER | CON_ENABLED,
    .index = -1,
};

/* ── flush workqueue ─────────────────────────────────────────────────── */

static void partlog_flush_work(struct work_struct *work)
{
    struct partlog_state *s =
        container_of(work, struct partlog_state, flush_work);
    char    tmp[PAGE_SIZE];
    size_t  n;
    unsigned long flags;
    u64 data_size = le64_to_cpu(s->hdr.data_size);
    u64 head;
    int ret;
    unsigned int chunks = 0;

    atomic_set(&s->flush_pending, 0);

    pr_debug(PARTLOG_TAG "flush_work enter (ring_used=%u)\n",
             s->ring.used);

    for (;;) {
        spin_lock_irqsave(&s->ring.lock, flags);
        n = ring_pop(&s->ring, tmp, PAGE_SIZE);
        spin_unlock_irqrestore(&s->ring.lock, flags);

        if (!n)
            break;

        head = le64_to_cpu(s->hdr.write_head);

        pr_debug(PARTLOG_TAG "chunk %u: %zu bytes, head=%llu/%llu\n",
                 chunks, n,
                 (unsigned long long)head,
                 (unsigned long long)data_size);

        if (head + n <= data_size) {
            ret = partlog_write(s,
                                (loff_t)(PARTLOG_DATA_START + head),
                                tmp, n);
            if (!ret)
                head += n;
        } else {
            /* wrap around the end of the data area */
            size_t p1 = (size_t)(data_size - head);
            size_t p2 = n - p1;

            pr_info(PARTLOG_TAG "partition wrap #%u: "
                    "%zu bytes at end, %zu at start\n",
                    s->wrap_count + 1, p1, p2);

            ret  = partlog_write(s,
                                 (loff_t)(PARTLOG_DATA_START + head),
                                 tmp, p1);
            ret |= partlog_write(s,
                                 (loff_t)PARTLOG_DATA_START,
                                 tmp + p1, p2);
            if (!ret) {
                head = p2;
                s->wrap_count++;
            }
        }

        if (ret) {
            pr_err(PARTLOG_TAG "flush error %d — %zu bytes lost "
                   "(head=%llu)\n",
                   ret, n, (unsigned long long)head);
            break;
        }

        s->total_written += n;
        s->hdr.write_head = cpu_to_le64(head);

        ret = partlog_write_header(s);
        if (ret)
            pr_err(PARTLOG_TAG "header sync failed after chunk %u\n", chunks);

        chunks++;
    }

    if (chunks)
        pr_debug(PARTLOG_TAG "flush_work done: %u chunks, "
                 "lifetime_written=%llu bytes\n",
                 chunks, s->total_written);
}

/* ── panic notifier ──────────────────────────────────────────────────── */

static int partlog_panic_notify(struct notifier_block *nb,
                                 unsigned long action, void *data)
{
    struct partlog_state *s = g_state;

    if (!s || !atomic_read(&s->initialised))
        return NOTIFY_DONE;

    /*
     * pr_emerg goes through the panic console chain which may or may
     * not reach serial/earlycon — emit it anyway for any console still
     * alive at this point.
     */
    pr_emerg(PARTLOG_TAG "panic detected — flushing log partition\n");

    partlog_flush_work(&s->flush_work);

    pr_emerg(PARTLOG_TAG "flush done (written=%llu drops=%llu wraps=%u) "
             "— issuing storage flush\n",
             s->total_written,
             s->ring.total_dropped,
             s->wrap_count);

    blkdev_issue_flush(s->bdev, GFP_NOIO, NULL);

    pr_emerg(PARTLOG_TAG "storage flush complete\n");
    return NOTIFY_OK;
}

static struct notifier_block partlog_panic_nb = {
    .notifier_call = partlog_panic_notify,
    .priority      = INT_MAX,
};

/* ── partition label helper ──────────────────────────────────────────── */

static const char *pl_get_label(struct hd_struct *part)
{

    if (part->info){
        return part->info->volname;
        }else {
            pr_info(PARTLOG_TAG "part %d has no info struct\n", part->partno);
    return NULL;
    }
}

/* ── partition matching ──────────────────────────────────────────────── */

static bool partlog_match(struct hd_struct *part, struct gendisk *disk)
{
    const char *lbl = pl_get_label(part);

    /* label match (preferred) */
    if (partlog_label[0]) {
        if (lbl && !strcmp(lbl, partlog_label)) {
            pr_debug(PARTLOG_TAG "label match: %s%d label='%s'\n",
                     disk->disk_name, part->partno, lbl);
            return true;
        }
    }

    /* disk+part_nr fallback */
    if (partlog_disk[0] && partlog_part_nr) {
        if (!strcmp(disk->disk_name, partlog_disk) &&
            (unsigned int)part->partno == partlog_part_nr) {
            pr_debug(PARTLOG_TAG "disk/part match: %s%d\n",
                     disk->disk_name, part->partno);
            return true;
        }
    }

    return false;
}

/* ── attach once a matching bdev is found ────────────────────────────── */

static void partlog_attach(struct hd_struct *part, struct gendisk *disk)
{
    struct partlog_state *s = g_state;
    struct block_device  *bdev;
    const char *lbl = pl_get_label(part);
    u64 data_size;
    int ret;

    pr_info(PARTLOG_TAG "attaching to %s%d (label='%s', size=%llu sectors)\n",
            disk->disk_name, part->partno,
            lbl ?: "<none>",
            (unsigned long long)part->nr_sects);

    bdev = bdget_disk(disk, part->partno);
    if (!bdev) {
        pr_err(PARTLOG_TAG "bdget_disk(%s, %d) returned NULL\n",
               disk->disk_name, part->partno);
        return;
    }

    ret = blkdev_get(bdev, FMODE_READ | FMODE_WRITE, NULL);
    if (ret) {
        pr_err(PARTLOG_TAG "blkdev_get failed: %d\n", ret);
        bdput(bdev);
        return;
    }

    pr_debug(PARTLOG_TAG "blkdev_get OK, bdev=%p\n", bdev);

    s->bdev = bdev;
    data_size = (u64)partlog_size_mb * 1024ULL * 1024ULL - PARTLOG_DATA_START;

    pr_info(PARTLOG_TAG "data area: %llu bytes (%u MiB - %d header bytes)\n",
            (unsigned long long)data_size,
            partlog_size_mb, PARTLOG_DATA_START);

    /* read existing header */
    ret = partlog_read(s, 0, &s->hdr, sizeof(s->hdr));
    if (ret) {
        pr_err(PARTLOG_TAG "header read failed: %d\n", ret);
        goto err;
    }

    pr_debug(PARTLOG_TAG "raw header: magic=0x%08X version=%u "
             "boot_count=%llu write_head=%llu\n",
             le32_to_cpu(s->hdr.magic),
             le32_to_cpu(s->hdr.version),
             (unsigned long long)le64_to_cpu(s->hdr.boot_count),
             (unsigned long long)le64_to_cpu(s->hdr.write_head));

    if (le32_to_cpu(s->hdr.magic)   != PARTLOG_MAGIC ||
        le32_to_cpu(s->hdr.version) != PARTLOG_VERSION) {
        pr_info(PARTLOG_TAG "no valid header found (magic=0x%08X) — "
                "formatting fresh log partition\n",
                le32_to_cpu(s->hdr.magic));
        memset(&s->hdr, 0, sizeof(s->hdr));
        s->hdr.magic      = cpu_to_le32(PARTLOG_MAGIC);
        s->hdr.version    = cpu_to_le32(PARTLOG_VERSION);
        s->hdr.boot_count = cpu_to_le64(1);
        s->hdr.write_head = cpu_to_le64(0);
        s->hdr.data_size  = cpu_to_le64(data_size);
    } else {
        u64 bc   = le64_to_cpu(s->hdr.boot_count);
        u64 wh   = le64_to_cpu(s->hdr.write_head);
        u64 used = (wh * 100) / data_size;

        pr_info(PARTLOG_TAG "existing log found — boot #%llu, "
                "head=%llu (%llu%% full)\n",
                (unsigned long long)bc,
                (unsigned long long)wh,
                (unsigned long long)used);

        s->hdr.boot_count = cpu_to_le64(bc + 1);
        s->hdr.data_size  = cpu_to_le64(data_size);
    }

    ret = partlog_write_header(s);
    if (ret) {
        pr_err(PARTLOG_TAG "initial header write failed: %d — aborting\n", ret);
        goto err;
    }

    atomic_set(&s->initialised, 1);

    /* CON_PRINTBUFFER replays the kernel log ring into our .write callback */
    register_console(&partlog_console);
    atomic_notifier_chain_register(&panic_notifier_list, &partlog_panic_nb);

    pr_info(PARTLOG_TAG "*** logging active *** "
            "partition=%s%d label=%s size=%uMiB boot=%llu\n",
            disk->disk_name, part->partno,
            lbl ?: "?",
            partlog_size_mb,
            (unsigned long long)le64_to_cpu(s->hdr.boot_count));
    return;

err:
    pr_err(PARTLOG_TAG "attach failed — logging disabled\n");
    blkdev_put(bdev, FMODE_READ | FMODE_WRITE);
    s->bdev = NULL;
}

/* ── class_interface ─────────────────────────────────────────────────── */

extern struct class       block_class;
extern struct device_type part_type;

static void pl_iface_add(struct device *dev, struct class_interface *ci)
{
    struct hd_struct *part;
    struct gendisk   *disk;

    if (dev->type != &part_type)
        return;

    part = dev_to_part(dev);
    disk = part_to_disk(part);

    pr_debug(PARTLOG_TAG "add_dev: %s%d (label='%s')\n",
             disk->disk_name, part->partno,
             pl_get_label(part) ?: "");

    if (!partlog_match(part, disk))
        return;

    mutex_lock(&partlog_open_mutex);
    if (!atomic_read(&g_state->initialised))
        partlog_attach(part, disk);
    else
        pr_debug(PARTLOG_TAG "already attached, ignoring %s%d\n",
                 disk->disk_name, part->partno);
    mutex_unlock(&partlog_open_mutex);
}

static void pl_iface_remove(struct device *dev, struct class_interface *ci)
{
    /* We hold a blkdev_get reference — nothing to do on removal. */
    pr_debug(PARTLOG_TAG "remove_dev called (ignored)\n");
}

static struct class_interface partlog_class_iface = {
    .add_dev    = pl_iface_add,
    .remove_dev = pl_iface_remove,
};

/* ── init / exit ─────────────────────────────────────────────────────── */

static int __init partlog_init(void)
{
    struct partlog_state *s;
    int ret;

    pr_info(PARTLOG_TAG "initialising (label='%s' disk='%s' part=%u size=%uMiB)\n",
            partlog_label, partlog_disk, partlog_part_nr, partlog_size_mb);

    if (!partlog_label[0] && (!partlog_disk[0] || !partlog_part_nr)) {
        pr_err(PARTLOG_TAG "no partition specified — "
               "set label= or disk=+part_nr= params\n");
        return -EINVAL;
    }

    s = kzalloc(sizeof(*s), GFP_KERNEL);
    if (!s) {
        pr_err(PARTLOG_TAG "failed to allocate driver state\n");
        return -ENOMEM;
    }

    spin_lock_init(&s->ring.lock);
    INIT_WORK(&s->flush_work, partlog_flush_work);
    atomic_set(&s->initialised, 0);
    atomic_set(&s->flush_pending, 0);

    s->io_buf = (void *)__get_free_page(GFP_KERNEL);
    if (!s->io_buf) {
        pr_err(PARTLOG_TAG "failed to allocate I/O bounce buffer\n");
        kfree(s);
        return -ENOMEM;
    }

    s->wq = alloc_workqueue("partlog", WQ_UNBOUND | WQ_MEM_RECLAIM, 1);
    if (!s->wq) {
        pr_err(PARTLOG_TAG "failed to create workqueue\n");
        free_page((unsigned long)s->io_buf);
        kfree(s);
        return -ENOMEM;
    }

    g_state = s;

    partlog_class_iface.class = &block_class;
    ret = class_interface_register(&partlog_class_iface);
    if (ret) {
        pr_err(PARTLOG_TAG "class_interface_register failed: %d\n", ret);
        destroy_workqueue(s->wq);
        free_page((unsigned long)s->io_buf);
        kfree(s);
        g_state = NULL;
        return ret;
    }

    if (atomic_read(&s->initialised))
        pr_info(PARTLOG_TAG "partition found and attached during init\n");
    else
        pr_info(PARTLOG_TAG "waiting for eMMC partition to come online...\n");

    return 0;
}

static void __exit partlog_exit(void)
{
    struct partlog_state *s = g_state;

    pr_info(PARTLOG_TAG "shutting down\n");

    class_interface_unregister(&partlog_class_iface);

    if (!s)
        return;

    if (atomic_read(&s->initialised)) {
        pr_info(PARTLOG_TAG "final stats: written=%llu bytes "
                "dropped=%llu bytes wraps=%u\n",
                s->total_written,
                s->ring.total_dropped,
                s->wrap_count);

        atomic_notifier_chain_unregister(&panic_notifier_list, &partlog_panic_nb);
        unregister_console(&partlog_console);
        flush_workqueue(s->wq);
        pr_info(PARTLOG_TAG "flush complete\n");
    } else {
        pr_warn(PARTLOG_TAG "exiting without ever attaching to a partition\n");
    }

    destroy_workqueue(s->wq);

    if (s->bdev) {
        blkdev_issue_flush(s->bdev, GFP_KERNEL, NULL);
        blkdev_put(s->bdev, FMODE_READ | FMODE_WRITE);
        pr_info(PARTLOG_TAG "block device released\n");
    }

    free_page((unsigned long)s->io_buf);
    kfree(s);
    g_state = NULL;

    pr_info(PARTLOG_TAG "shutdown complete\n");
}

late_initcall(partlog_init);
module_exit(partlog_exit);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("Persistent kernel log to raw partition (no pstore/udev needed)");
MODULE_VERSION("2.1");