import { themeConfig } from '@/config'
import type { DateFormat } from '@/types'

const MONTHS_EN = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
]

const VALID_SEPARATORS = ['.', '-', '/']

/**
 * @param date
 * @param format
 * @returns
 */
export function formatDate(date: Date, format?: string): string {
  const formatStr = (format || themeConfig.date.dateFormat).trim()
  const configSeparator = themeConfig.date.dateSeparator || '-'

  const separator = VALID_SEPARATORS.includes(configSeparator.trim()) ? configSeparator.trim() : '.'

  // Frontmatter dates are coerced to UTC midnight, so read them back in UTC.
  // Local getters shift the day back by one in any timezone behind UTC.
  const year = date.getUTCFullYear()
  const month = date.getUTCMonth() + 1
  const day = date.getUTCDate()
  const monthName = MONTHS_EN[date.getUTCMonth()]

  const pad = (num: number) => String(num).padStart(2, '0')

  switch (formatStr) {
    case 'YYYY-MM-DD':
      return `${year}${separator}${pad(month)}${separator}${pad(day)}`

    case 'MM-DD-YYYY':
      return `${pad(month)}${separator}${pad(day)}${separator}${year}`

    case 'DD-MM-YYYY':
      return `${pad(day)}${separator}${pad(month)}${separator}${year}`

    case 'MONTH DAY YYYY':
      return `<span class="month">${monthName}</span> ${day} ${year}`

    case 'DAY MONTH YYYY':
      return `${day} <span class="month">${monthName}</span> ${year}`

    default:
      return `${year}${separator}${pad(month)}${separator}${pad(day)}`
  }
}

export const SUPPORTED_DATE_FORMATS: readonly DateFormat[] = [
  'YYYY-MM-DD',
  'MM-DD-YYYY',
  'DD-MM-YYYY',
  'MONTH DAY YYYY',
  'DAY MONTH YYYY'
] as const
